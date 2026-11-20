"""Synopsis against narrative on the other ASRS primary-problem tasks. For
every single primary-problem category other than Aircraft, builds the
balanced binary task with the rules of asrs_pipeline/build_task.py (reports
through 2021 with a primary problem and a narrative, first occurrence per
ACN, the larger class undersampled to 50:50, a stratified 80/20 split) and
scores both records with one model: word TF-IDF with logistic regression
(one fit), or the BiLSTM over frozen GloVe-200 (three trainings, tasks of
400 records or more). The interval is a joint case bootstrap of 2000
resamples. Reads the export in ASRS_DIR and, for the BiLSTM,
glove.6B.200d.txt in EMB_DIR; appends one line per task and model to
other_tasks.jsonl and skips tasks already there.
Run: ASRS_DIR=<csv export> python3 other_tasks.py tfidf|bilstm [max_task_size]"""
import csv
import glob
import json
import os
import re
import sys
import time
from collections import Counter

import numpy as np

csv.field_size_limit(10**7)

HERE = os.path.dirname(os.path.abspath(__file__))
ASRS_DIR = os.environ['ASRS_DIR']
EMB_DIR = os.environ.get('EMB_DIR', os.path.join(os.path.dirname(HERE), 'embeddings'))
OUT = os.path.join(HERE, 'other_tasks.jsonl')
TOKEN_RE = re.compile(r"[a-z][a-z0-9/-]+")
SEED = 20260802
MAX_YEAR = 2021
PAD = 0
OOV = 1
CAPS = {'narr': 256, 'syn': 64}
B = 2000


def read_synopses():
    synopses = {}
    for path in sorted(glob.glob(os.path.join(ASRS_DIR, '*.csv'))):
        with open(path, errors='replace') as f:
            reader = csv.reader(f)
            try:
                header1 = next(reader)
                header2 = next(reader)
            except StopIteration:
                continue
            columns = [f"{a}/{b}".strip('/') for a, b in zip(header1, header2)]
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
    return synopses


def read_pool():
    """Every report through MAX_YEAR with a primary problem and a narrative,
    first occurrence per ACN, with its synopsis."""
    synopses = read_synopses()
    seen = set()
    pool = []
    for name in sorted(f for f in os.listdir(ASRS_DIR) if f.endswith('.csv')):
        with open(os.path.join(ASRS_DIR, name), newline='', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f)
            try:
                sections = next(reader)
                fields = next(reader)
            except StopIteration:
                continue
            header = [(s.strip(), t.strip()) for s, t in zip(sections, fields)]
            columns = {}
            for key, section, field in [('date', 'Time', 'Date'), ('primary', 'Assessments', 'Primary Problem'),
                                        ('narrative', 'Report 1', 'Narrative')]:
                hits = [i for i, (s, t) in enumerate(header) if s == section and t == field]
                if not hits and key == 'primary':
                    hits = [i for i, (_, t) in enumerate(header) if t == 'Primary Problem']
                columns[key] = hits[0] if hits else None
            if None in columns.values():
                continue
            for row in reader:
                if len(row) <= columns['narrative'] or not row[0].strip().isdigit():
                    continue
                date = row[columns['date']].strip()
                if not re.fullmatch(r"(19|20)\d{4}", date):
                    continue
                if int(date) // 100 > MAX_YEAR:
                    continue
                acn = row[0].strip()
                primary = row[columns['primary']].strip()
                narrative = row[columns['narrative']].strip()
                if acn in seen or not primary or not narrative:
                    continue
                seen.add(acn)
                pool.append({'acn': acn, 'primary': primary, 'narr': narrative, 'syn': synopses.get(acn, '')})
    return pool


def build_task(pool, category):
    """Balanced task, positives first, shuffled, then a stratified 80/20 split."""
    rng = np.random.default_rng(SEED)
    positives = [dict(r, label=1) for r in pool if r['primary'] == category]
    negatives = [dict(r, label=0) for r in pool if r['primary'] != category]
    n_category = len(positives)
    if len(negatives) >= len(positives):
        keep = rng.choice(len(negatives), size=len(positives), replace=False)
        negatives = [negatives[i] for i in sorted(keep)]
    else:
        keep = rng.choice(len(positives), size=len(negatives), replace=False)
        positives = [positives[i] for i in sorted(keep)]
    data = positives + negatives
    order = rng.permutation(len(data))
    data = [data[i] for i in order]
    test = set()
    for label in (0, 1):
        acns = [r['acn'] for r in data if r['label'] == label]
        shuffled = rng.permutation(len(acns))
        n_test = round(0.2 * len(acns))
        test.update(acns[i] for i in shuffled[:n_test])
    y = np.array([r['label'] for r in data], np.int64)
    is_test = np.array([r['acn'] in test for r in data])
    return data, y, is_test, n_category


def macro_f1(y, pred):
    """Binary macro-F1 from counts; equals f1_score(average='macro')."""
    scores = []
    for c in (0, 1):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return float(np.mean(scores))


def joint_ci(y, synopsis_runs, narrative_runs, seed):
    n = len(y)
    rng = np.random.default_rng(SEED + seed)
    boot = np.empty(B)
    for i in range(B):
        index = rng.integers(0, n, n)
        boot[i] = np.mean([macro_f1(y[index], s[index]) - macro_f1(y[index], q[index])
                           for s, q in zip(synopsis_runs, narrative_runs)])
    return [round(float(v), 4) for v in np.percentile(boot, [2.5, 97.5])]


def done():
    if not os.path.exists(OUT):
        return set()
    rows = [json.loads(line) for line in open(OUT)]
    return {(r['category'], r['model']) for r in rows}


def emit(row):
    with open(OUT, 'a') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


def score_tfidf(data, y, is_test, row):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    y_test = y[is_test]
    preds = {}
    for record in ('narr', 'syn'):
        texts = [r[record] for r in data]
        vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
        try:
            X_train = vectorizer.fit_transform([t for t, train in zip(texts, ~is_test) if train])
            X_test = vectorizer.transform([t for t, test in zip(texts, is_test) if test])
            model = LogisticRegression(max_iter=2000, C=1.0).fit(X_train, y[~is_test])
            preds[record] = (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
        except ValueError as error:
            preds[record] = None
            row['error'] = str(error)
    if preds['narr'] is not None and preds['syn'] is not None:
        row['narr_f1'] = [round(macro_f1(y_test, preds['narr']), 4)]
        row['syn_f1'] = [round(macro_f1(y_test, preds['syn']), 4)]
        row['delta_mean'] = round(row['syn_f1'][0] - row['narr_f1'][0], 4)
        row['ci95'] = joint_ci(y_test, [preds['syn']], [preds['narr']], 800)


def load_glove():
    glove = {}
    for line in open(os.path.join(EMB_DIR, 'glove.6B.200d.txt'), encoding='utf-8'):
        parts = line.rstrip().split(' ')
        glove[parts[0]] = np.asarray(parts[1:], np.float32)
    return glove


def score_bilstm(data, y, is_test, row, glove):
    import torch
    import torch.nn as nn
    from sklearn.model_selection import train_test_split
    torch.set_num_threads(1)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    class BiLSTM(nn.Module):
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
                logits = model(torch.from_numpy(X[start:start + 256]).long().to(device))
                outputs.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
        return np.concatenate(outputs)

    y_test = y[is_test]
    runs = {'narr': [], 'syn': []}
    for record in ('narr', 'syn'):
        cap = CAPS[record]
        tokens = [TOKEN_RE.findall(r[record].lower())[:cap] for r in data]
        vocab = set(t for row_tokens in tokens for t in row_tokens)
        rng = np.random.default_rng(0)
        index = {}
        vectors = [np.zeros(200, np.float32), rng.normal(0, 0.1, 200).astype(np.float32)]
        # rows in GloVe file order
        for word in glove:
            if word in vocab:
                index[word] = len(vectors)
                vectors.append(glove[word])
        matrix = np.stack(vectors)
        X = np.zeros((len(tokens), cap), np.int32)
        for i, row_tokens in enumerate(tokens):
            for j, token in enumerate(row_tokens):
                X[i, j] = index.get(token, OOV)
        for seed in (0, 1, 2):
            X_train, X_val, y_train, y_val = train_test_split(X[~is_test], y[~is_test], test_size=0.05,
                                                              stratify=y[~is_test], random_state=SEED + seed)
            torch.manual_seed(100 + seed)
            model = BiLSTM(matrix).to(device)
            optimiser = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
            loss_fn = nn.CrossEntropyLoss()
            X_tensor = torch.from_numpy(X_train).long()
            y_tensor = torch.from_numpy(y_train)
            best = -1.0
            best_state = None
            patience = 0
            for epoch in range(15):
                model.train()
                generator = torch.Generator().manual_seed((100 + seed) * 1000 + epoch)
                perm = torch.randperm(len(X_tensor), generator=generator)
                for start in range(0, len(perm), 128):
                    batch = perm[start:start + 128]
                    optimiser.zero_grad()
                    loss_fn(model(X_tensor[batch].to(device)), y_tensor[batch].to(device)).backward()
                    optimiser.step()
                f1 = macro_f1(y_val, (predict(model, X_val) >= 0.5).astype(int))
                if f1 > best + 1e-4:
                    best = f1
                    patience = 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience += 1
                    if patience >= 2:
                        break
            if best_state:
                model.load_state_dict(best_state)
            runs[record].append((predict(model, X[is_test]) >= 0.5).astype(int))
    row['narr_f1'] = [round(macro_f1(y_test, p), 4) for p in runs['narr']]
    row['syn_f1'] = [round(macro_f1(y_test, p), 4) for p in runs['syn']]
    row['delta_per_training'] = [round(a - b, 4) for a, b in zip(row['syn_f1'], row['narr_f1'])]
    row['delta_mean'] = round(float(np.mean([macro_f1(y_test, s) - macro_f1(y_test, q)
                                             for s, q in zip(runs['syn'], runs['narr'])])), 4)
    row['ci95'] = joint_ci(y_test, runs['syn'], runs['narr'], 801)


def main():
    model_name = sys.argv[1]
    max_size = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
    pool = read_pool()
    counts = Counter(r['primary'] for r in pool)
    categories = [c for c, _ in sorted(counts.items(), key=lambda kv: kv[1]) if ';' not in c and c != 'Aircraft']
    print('pool', len(pool), 'single categories', len(categories) + 1, {c: counts[c] for c in categories}, flush=True)
    finished = done()
    glove = load_glove() if model_name == 'bilstm' else None
    for category in categories:
        if (category, model_name) in finished:
            continue
        data, y, is_test, n_category = build_task(pool, category)
        if len(data) > max_size:
            print('skip (size)', category, len(data), flush=True)
            continue
        start = time.time()
        row = {'category': category, 'model': model_name, 'n_category_records': n_category, 'n_task': len(data),
               'n_heldout': int(is_test.sum()),
               'synopsis_blank_heldout': int(sum(1 for r, test in zip(data, is_test) if test and not r['syn']))}
        if model_name == 'tfidf':
            score_tfidf(data, y, is_test, row)
        elif len(data) < 400:
            row['skipped'] = 'balanced task under 400 records'
            emit(row)
            continue
        else:
            score_bilstm(data, y, is_test, row, glove)
        row['secs'] = round(time.time() - start)
        emit(row)
    print('done', model_name, flush=True)


if __name__ == '__main__':
    main()
