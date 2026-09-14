"""Trains the sixteen-class recall-campaign task with a split by campaign
year: campaigns filed 2000 to 2021 train, 2022 to 2026 test. Exact-duplicate
groups that span the boundary lose their test-side members. Per field it
trains TF-IDF with logistic regression and a BiLSTM over field-trained
word2vec, three seeds. Writes results/nhtsa/nhtsa_temporal.json.
Run: python3 nhtsa_temporal.py"""
import json
import os
import re
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RECORDS = os.path.join(ROOT, 'records')
NHTSA = os.path.join(ROOT, 'nhtsa')
RESULTS = os.path.join(ROOT, 'results')
TOKEN_RE = re.compile(r"[a-z][a-z0-9/-]+")
PAD, OOV = 0, 1
CAP = 96


def top_component(component):
    return (component or '').split(':')[0].split(',')[0].strip()


def build():
    seen = set()
    campaigns = []
    for line in open(f'{HERE}/nhtsa_campaigns.jsonl'):
        record = json.loads(line)
        if record['NHTSACampaignNumber'] in seen:
            continue
        seen.add(record['NHTSACampaignNumber'])
        campaigns.append(record)
    from collections import Counter
    class_counts = Counter(top_component(record['Component']) for record in campaigns)
    classes = sorted([name for name, count in class_counts.items() if count >= 300 and name])
    label_index = {name: i for i, name in enumerate(classes)}
    data = [record for record in campaigns if top_component(record['Component']) in label_index]
    for record in data:
        # the first two digits of a campaign number are the filing year
        record['year'] = 2000 + int(record['NHTSACampaignNumber'][:2])
        record['split'] = 'test' if record['year'] >= 2022 else 'train'
    parent = list(range(len(data)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for field in ('Summary', 'Consequence', 'Remedy'):
        first_seen = {}
        for i, record in enumerate(data):
            key = (record[field] or '').strip().lower()[:400]
            if not key:
                continue
            if key in first_seen:
                union(first_seen[key], i)
            else:
                first_seen[key] = i
    groups = {}
    for i, record in enumerate(data):
        groups.setdefault(find(i), []).append(i)
    dropped = set()
    for group in groups.values():
        sides = {data[i]['split'] for i in group}
        if len(sides) > 1:
            dropped.update(i for i in group if data[i]['split'] == 'test')
    data = [record for i, record in enumerate(data) if i not in dropped]
    y = np.array([label_index[top_component(record['Component'])] for record in data])
    is_test = np.array([record['split'] == 'test' for record in data])
    print(f"temporal task {len(data)} (dropped {len(dropped)} boundary-dup test campaigns), test {int(is_test.sum())}", flush=True)
    return data, y, is_test, len(classes)


def main():
    import torch
    import torch.nn as nn
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from gensim.models import Word2Vec
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    data, y, is_test, n_classes = build()
    out = {"n": len(data), "n_test": int(is_test.sum())}

    class RNN(nn.Module):
        def __init__(self, matrix, n_classes):
            super().__init__()
            self.emb = nn.Embedding.from_pretrained(torch.from_numpy(matrix), freeze=True, padding_idx=PAD)
            self.rnn = nn.LSTM(matrix.shape[1], 64, batch_first=True, bidirectional=True)
            self.drop = nn.Dropout(0.3)
            self.fc = nn.Linear(128, n_classes)

        def forward(self, x):
            hidden = self.rnn(self.emb(x))[1][0]
            return self.fc(self.drop(torch.cat([hidden[0], hidden[1]], dim=1)))

    def predict(model, X):
        model.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(X), 256):
                outputs.append(model(torch.from_numpy(X[start:start + 256]).long().to(device)).argmax(1).cpu().numpy())
        return np.concatenate(outputs)

    for field_key, field in (('summary', 'Summary'), ('conseq', 'Consequence'), ('remedy', 'Remedy')):
        texts = [(record[field] or '') for record in data]
        vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
        Xtr = vectorizer.fit_transform([t for t, m in zip(texts, ~is_test) if m])
        Xte = vectorizer.transform([t for t, m in zip(texts, is_test) if m])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, y[~is_test])
        out[f'tfidf_{field_key}'] = round(float(f1_score(y[is_test], clf.predict(Xte), average='macro')), 4)
        print('tfidf', field_key, out[f'tfidf_{field_key}'], flush=True)
        sentences = [TOKEN_RE.findall(t.lower()) for t, m in zip(texts, ~is_test) if m and t.strip()]
        w2v = Word2Vec(vector_size=200, window=5, min_count=3, sg=1, epochs=10, workers=8, seed=20260802)
        w2v.build_vocab(sentences)
        w2v.train(corpus_iterable=sentences, total_examples=len(sentences), epochs=10)
        token_lists = [TOKEN_RE.findall(t.lower())[:CAP] for t in texts]
        vocab = set(t for ts in token_lists for t in ts)
        oov_rng = np.random.default_rng(0)
        word_index, vectors = {}, [np.zeros(200, np.float32), oov_rng.normal(0, 0.1, 200).astype(np.float32)]
        for word in sorted(vocab):
            if word in w2v.wv:
                word_index[word] = len(vectors)
                vectors.append(w2v.wv[word].astype(np.float32))
        matrix = np.stack(vectors)
        del w2v
        X = np.zeros((len(token_lists), CAP), np.int32)
        for i, ts in enumerate(token_lists):
            for j, t in enumerate(ts):
                X[i, j] = word_index.get(t, OOV)
        Xtr_all, ytr_all = X[~is_test], y[~is_test]
        Xte_, yte_ = X[is_test], y[is_test]
        for seed in (0, 1, 2):
            Xtr2, Xva, ytr2, yva = train_test_split(Xtr_all, ytr_all, test_size=0.05, stratify=ytr_all, random_state=20260802 + seed)
            torch.manual_seed(100 + seed)
            model = RNN(matrix, n_classes).to(device)
            optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
            loss_fn = nn.CrossEntropyLoss()
            Xtr_t = torch.from_numpy(Xtr2).long()
            ytr_t = torch.from_numpy(ytr2)
            best_f1, best_state, patience = -1.0, None, 0
            for epoch in range(15):
                model.train()
                perm = torch.randperm(len(Xtr_t), generator=torch.Generator().manual_seed((100 + seed) * 1000 + epoch))
                for start in range(0, len(perm), 128):
                    batch_idx = perm[start:start + 128]
                    optimizer.zero_grad()
                    loss = loss_fn(model(Xtr_t[batch_idx].to(device)), ytr_t[batch_idx].to(device))
                    loss.backward()
                    optimizer.step()
                f1 = f1_score(yva, predict(model, Xva), average='macro')
                if f1 > best_f1 + 1e-4:
                    best_f1, patience = f1, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience += 1
                    if patience >= 2:
                        break
            if best_state:
                model.load_state_dict(best_state)
            out[f'bilstm_{field_key}_s{seed}'] = round(float(f1_score(yte_, predict(model, Xte_), average='macro')), 4)
            print('bilstm', field_key, seed, out[f'bilstm_{field_key}_s{seed}'], flush=True)
            del model
        del matrix, X, Xtr_t
        json.dump(out, open(f'{RESULTS}/nhtsa/nhtsa_temporal.json', 'w'), indent=1)
    json.dump(out, open(f'{RESULTS}/nhtsa/nhtsa_temporal.json', 'w'), indent=1)
    print("TEMPORAL DONE", flush=True)


if __name__ == '__main__':
    main()
