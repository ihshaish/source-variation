"""Redraws the negative class and reruns the record contrast, one draw
per invocation. Draw d resamples the negatives with seed 100+d from the
same pool, Aircraft positives fixed, with the same 80/20 stratified split
policy, then trains word TF-IDF and a BiLSTM over draw-trained word2vec
on narrative and synopsis, one training each, with the same constants as
build_task.py, records_word2vec.py and records_train.py. The statistic is
synopsis minus narrative macro-F1 per model per draw, appended to
sensitivity_draws.json. Run python3 sensitivity_draws.py 0 first to build
pool_records.jsonl.gz from the CSV export in ASRS_DIR (107 MB, not
shipped), then python3 sensitivity_draws.py <draw 1..5>."""
import csv
import glob
import gzip
import json
import os
import re
import sys
import numpy as np

csv.field_size_limit(10**7)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RECORDS = os.path.join(ROOT, 'records')
NHTSA = os.path.join(ROOT, 'nhtsa')
RESULTS = os.path.join(ROOT, 'results')
POOL = os.path.join(HERE, 'pool_records.jsonl.gz')
TOKEN_RE = re.compile(r"[a-z][a-z0-9/-]+")
MAX_YEAR = 2021
PAD, OOV = 0, 1
CAPS = {'narr': 256, 'syn': 64}


def build_pool():
    synopses = {}
    for path in sorted(glob.glob(os.path.join(os.environ['ASRS_DIR'], '*.csv'))):
        with open(path, errors='replace') as f:
            reader = csv.reader(f)
            try:
                header1 = next(reader)
                header2 = next(reader)
            except StopIteration:
                continue
            columns = [f"{a.strip()}/{b.strip()}" for a, b in zip(header1, header2)]
            try:
                acn_col = [i for i, c in enumerate(columns) if c.endswith('ACN')][0]
                syn_col = columns.index('Report 1/Synopsis')
            except (ValueError, IndexError):
                continue
            for row in reader:
                if len(row) <= max(acn_col, syn_col):
                    continue
                acn = row[acn_col].strip()
                if acn:
                    synopses[acn] = row[syn_col].strip()
    seen = set()
    count = 0
    with gzip.open(POOL, 'wt') as out:
        for name in sorted(f for f in os.listdir(os.environ['ASRS_DIR']) if f.endswith('.csv')):
            with open(os.path.join(os.environ['ASRS_DIR'], name), newline='', encoding='utf-8', errors='replace') as f:
                reader = csv.reader(f)
                try:
                    sections = next(reader)
                    fields = next(reader)
                except StopIteration:
                    continue
                header = [(s.strip(), t.strip()) for s, t in zip(sections, fields)]
                column = {}
                for key, section, field in [("date", "Time", "Date"), ("primary", "Assessments", "Primary Problem"),
                                            ("narrative", "Report 1", "Narrative")]:
                    hits = [i for i, (s, t) in enumerate(header) if s == section and t == field]
                    if not hits and key == "primary":
                        hits = [i for i, (_, t) in enumerate(header) if t == "Primary Problem"]
                    column[key] = hits[0] if hits else None
                if None in column.values():
                    continue
                for row in reader:
                    if len(row) <= column["narrative"] or not row[0].strip().isdigit():
                        continue
                    date = row[column["date"]].strip()
                    if not re.fullmatch(r"(19|20)\d{4}", date):
                        continue
                    year = int(date) // 100
                    if year > MAX_YEAR:
                        continue
                    acn = row[0].strip()
                    primary = row[column["primary"]].strip()
                    narr = row[column["narrative"]].strip()
                    if acn in seen or not primary or not narr:
                        continue
                    seen.add(acn)
                    count += 1
                    out.write(json.dumps({"acn": acn, "label": 1 if primary == "Aircraft" else 0,
                                          "narr": narr, "syn": synopses.get(acn, '')}) + "\n")
    print("pool records:", count, flush=True)


def run_draw(draw):
    import torch
    import torch.nn as nn
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from gensim.models import Word2Vec
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    records = [json.loads(line) for line in gzip.open(POOL, 'rt')]
    positives = [r for r in records if r['label'] == 1]
    negatives = [r for r in records if r['label'] == 0]
    rng = np.random.default_rng(100 + draw)
    keep = rng.choice(len(negatives), size=len(positives), replace=False)
    data = positives + [negatives[i] for i in sorted(keep)]
    order = rng.permutation(len(data))
    data = [data[i] for i in order]
    test = set()
    for label in (0, 1):
        class_acns = [r['acn'] for r in data if r['label'] == label]
        class_order = rng.permutation(len(class_acns))
        n_test = round(0.2 * len(class_acns))
        test.update(class_acns[i] for i in class_order[:n_test])
    for r in data:
        r['split'] = 'test' if r['acn'] in test else 'train'
    print(f"draw {draw}: task {len(data)} test {len(test)}", flush=True)
    y = np.array([r['label'] for r in data], np.int64)
    is_test = np.array([r['split'] == 'test' for r in data])
    result = {"draw": draw, "n": len(data), "n_test": int(is_test.sum())}
    for record in ('narr', 'syn'):
        texts = [r[record] for r in data]
        vectoriser = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
        Xtr = vectoriser.fit_transform([t for t, m in zip(texts, ~is_test) if m])
        Xte = vectoriser.transform([t for t, m in zip(texts, is_test) if m])
        classifier = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, y[~is_test])
        pred = (classifier.predict_proba(Xte)[:, 1] >= 0.5).astype(int)
        result[f'tfidf_{record}'] = round(float(f1_score(y[is_test], pred, average='macro')), 4)
        print(draw, 'tfidf', record, result[f'tfidf_{record}'], flush=True)

    class RNN(nn.Module):
        def __init__(self, matrix):
            super().__init__()
            self.emb = nn.Embedding.from_pretrained(torch.from_numpy(matrix), freeze=True, padding_idx=PAD)
            self.rnn = nn.LSTM(matrix.shape[1], 64, batch_first=True, bidirectional=True)
            self.drop = nn.Dropout(0.3)
            self.fc = nn.Linear(128, 2)

        def forward(self, x):
            hidden = self.rnn(self.emb(x))[1][0]
            return self.fc(self.drop(torch.cat([hidden[0], hidden[1]], dim=1)))

    def predict(model, X):
        model.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(X), 256):
                outputs.append(torch.softmax(model(torch.from_numpy(X[start:start + 256]).long().to(device)), 1)[:, 1].cpu().numpy())
        return np.concatenate(outputs)

    for record in ('narr', 'syn'):
        cap = CAPS[record]
        sentences = [TOKEN_RE.findall(r[record].lower()) for r in data if r['split'] == 'train' and r[record].strip()]
        w2v = Word2Vec(vector_size=200, window=5, min_count=3, sg=1, epochs=10, workers=8, seed=20260802)
        w2v.build_vocab(sentences)
        w2v.train(corpus_iterable=sentences, total_examples=len(sentences), epochs=10)
        token_lists = [TOKEN_RE.findall(r[record].lower())[:cap] for r in data]
        vocab = set(t for ts in token_lists for t in ts)
        init_rng = np.random.default_rng(0)
        index = {}
        vectors = [np.zeros(200, np.float32), init_rng.normal(0, 0.1, 200).astype(np.float32)]
        for word in sorted(vocab):
            if word in w2v.wv:
                index[word] = len(vectors)
                vectors.append(w2v.wv[word].astype(np.float32))
        matrix = np.stack(vectors)
        del w2v
        X = np.zeros((len(token_lists), cap), np.int32)
        for i, tokens in enumerate(token_lists):
            for j, token in enumerate(tokens):
                X[i, j] = index.get(token, OOV)
        Xtr_all, ytr_all = X[~is_test], y[~is_test]
        X_test, y_test = X[is_test], y[is_test]
        Xtr, Xva, ytr, yva = train_test_split(Xtr_all, ytr_all, test_size=0.05, stratify=ytr_all, random_state=20260802)
        torch.manual_seed(100)
        model = RNN(matrix).to(device)
        optimiser = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        Xtr_t = torch.from_numpy(Xtr).long()
        ytr_t = torch.from_numpy(ytr)
        best, best_state, patience = -1.0, None, 0
        for epoch in range(15):
            model.train()
            perm = torch.randperm(len(Xtr_t), generator=torch.Generator().manual_seed(100000 + epoch))
            for start in range(0, len(perm), 128):
                batch = perm[start:start + 128]
                optimiser.zero_grad()
                loss = loss_fn(model(Xtr_t[batch].to(device)), ytr_t[batch].to(device))
                loss.backward()
                optimiser.step()
            f1 = f1_score(yva, (predict(model, Xva) >= 0.5).astype(int), average='macro')
            print(f"  d{draw} {record} epoch {epoch} val {f1:.4f}", flush=True)
            if f1 > best + 1e-4:
                best, patience = f1, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= 2:
                    break
        if best_state:
            model.load_state_dict(best_state)
        pred = (predict(model, X_test) >= 0.5).astype(int)
        result[f'bilstm_{record}'] = round(float(f1_score(y_test, pred, average='macro')), 4)
        print(draw, 'bilstm', record, result[f'bilstm_{record}'], flush=True)
        del model, matrix, X, Xtr_t
    result['tfidf_delta'] = round(result['tfidf_syn'] - result['tfidf_narr'], 4)
    result['bilstm_delta'] = round(result['bilstm_syn'] - result['bilstm_narr'], 4)
    results = []
    results_path = os.path.join(HERE, 'sensitivity_draws.json')
    if os.path.exists(results_path):
        results = json.load(open(results_path))
    results.append(result)
    json.dump(results, open(results_path, 'w'), indent=1)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    draw = int(sys.argv[1])
    if draw == 0:
        build_pool()
    else:
        run_draw(draw)
