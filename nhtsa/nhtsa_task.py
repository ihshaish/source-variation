"""Builds the NHTSA recall-campaign task from the crawled campaigns and trains
one classifier per field. Classes are the top-level component names with at
least 300 campaigns; campaigns sharing an identical summary, consequence or
remedy text are kept on one side of an 80/20 split. Per field it trains
word2vec, TF-IDF with logistic regression and a BiLSTM over three seeds, then
runs paired randomisation tests between fields with Holm correction. Writes
nhtsa_task_meta.json, nhtsa_task_results.json, nhtsa_task_contrasts.json and
one nhtsa_preds_*.npz per run. Run: python3 nhtsa_task.py"""
import json
import os
import re
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
TOKEN_RE = re.compile(r"[a-z][a-z0-9/-]+")
SEED = 20260802


def log(*args):
    print(*args, flush=True)


records = [json.loads(line) for line in open(os.path.join(HERE, 'nhtsa_campaigns.jsonl'))]
seen = set()
campaigns = []
for record in records:
    if record['NHTSACampaignNumber'] in seen:
        continue
    seen.add(record['NHTSACampaignNumber'])
    campaigns.append(record)
log("campaigns:", len(campaigns))
from collections import Counter


def top_component(component):
    return (component or '').split(':')[0].split(',')[0].strip()


class_counts = Counter(top_component(record['Component']) for record in campaigns)
classes = sorted([name for name, count in class_counts.items() if count >= 300 and name])
log("classes kept:", len(classes), classes)
label_index = {name: i for i, name in enumerate(classes)}
data = [record for record in campaigns if top_component(record['Component']) in label_index]
rng = np.random.default_rng(SEED)
# union-find over campaigns that share an identical text in any of the three fields
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
    groups.setdefault(find(i), []).append(record)
group_keys = list(groups)
rng.shuffle(group_keys)
n_test = int(0.2 * len(data))
test_campaigns = set()
count = 0
for key in group_keys:
    if count >= n_test:
        break
    for record in groups[key]:
        test_campaigns.add(record['NHTSACampaignNumber'])
    count += len(groups[key])
print("duplicate-union groups:", len(groups))
for record in data:
    record['split'] = 'test' if record['NHTSACampaignNumber'] in test_campaigns else 'train'
log("task:", len(data), "test", len(test_campaigns))
y = np.array([label_index[top_component(record['Component'])] for record in data])
is_test = np.array([record['split'] == 'test' for record in data])
json.dump({"classes": classes, "n": len(data), "n_test": int(is_test.sum())}, open(os.path.join(HERE, 'nhtsa_task_meta.json'), 'w'))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def macro_f1(labels, pred):
    return f1_score(labels, pred, average='macro')


FIELDS = {'summary': 'Summary', 'conseq': 'Consequence', 'remedy': 'Remedy'}
results = []
for field_key, field in FIELDS.items():
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    Xtr = vectorizer.fit_transform([record[field] or '' for record, m in zip(data, ~is_test) if m])
    Xte = vectorizer.transform([record[field] or '' for record, m in zip(data, is_test) if m])
    clf = LogisticRegression(max_iter=2000).fit(Xtr, y[~is_test])
    pred = clf.predict(Xte)
    probs = clf.predict_proba(Xte)
    np.savez(os.path.join(HERE, f'nhtsa_preds_{field_key}_tfidf_s0.npz'), pred=pred, probs=probs, y=y[is_test])
    results.append({"key": f"nhtsa_{field_key}_tfidf_s0", "f1": round(float(macro_f1(y[is_test], pred)), 4)})
    log(results[-1])
from gensim.models import Word2Vec
import torch
import torch.nn as nn
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
CAP = 96
PAD, OOV = 0, 1


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
        for start in range(0, len(X), 512):
            outputs.append(model(torch.from_numpy(X[start:start + 512]).long().to(device)).argmax(1).cpu().numpy())
    return np.concatenate(outputs)


for field_key, field in FIELDS.items():
    token_lists = [TOKEN_RE.findall((record[field] or '').lower())[:CAP] for record in data]
    sentences = [t for t, m in zip(token_lists, ~is_test) if m]
    w2v = Word2Vec(vector_size=200, window=5, min_count=3, sg=1, epochs=10, workers=8, seed=SEED)
    w2v.build_vocab(sentences)
    w2v.train(corpus_iterable=sentences, total_examples=len(sentences), epochs=10)
    vocab = set(t for ts in token_lists for t in ts)
    oov_rng = np.random.default_rng(0)
    word_index, vectors = {}, [np.zeros(200, np.float32), oov_rng.normal(0, 0.1, 200).astype(np.float32)]
    for word in sorted(vocab):
        if word in w2v.wv:
            word_index[word] = len(vectors)
            vectors.append(w2v.wv[word].astype(np.float32))
    matrix = np.stack(vectors)
    X = np.zeros((len(token_lists), CAP), np.int32)
    for i, ts in enumerate(token_lists):
        for j, t in enumerate(ts):
            X[i, j] = word_index.get(t, OOV)
    for seed in (0, 1, 2):
        Xtr, Xva, ytr, yva = train_test_split(X[~is_test], y[~is_test], test_size=0.05, stratify=y[~is_test], random_state=SEED + seed)
        torch.manual_seed(700 + seed)
        model = RNN(matrix, len(classes)).to(device)
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        Xtr_t = torch.from_numpy(Xtr).long()
        ytr_t = torch.from_numpy(ytr)
        best_f1, best_state, patience = -1, None, 0
        for epoch in range(15):
            model.train()
            perm = torch.randperm(len(Xtr_t), generator=torch.Generator().manual_seed(seed * 313 + epoch))
            for start in range(0, len(perm), 128):
                batch_idx = perm[start:start + 128]
                optimizer.zero_grad()
                loss_fn(model(Xtr_t[batch_idx].to(device)), ytr_t[batch_idx].to(device)).backward()
                optimizer.step()
            f1 = macro_f1(yva, predict(model, Xva))
            if f1 > best_f1 + 1e-4:
                best_f1, patience = f1, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= 2:
                    break
        model.load_state_dict(best_state)
        pred = predict(model, X[is_test])
        np.savez(os.path.join(HERE, f'nhtsa_preds_{field_key}_bilstm_s{seed}.npz'), pred=pred, y=y[is_test])
        results.append({"key": f"nhtsa_{field_key}_bilstm_s{seed}", "f1": round(float(macro_f1(y[is_test], pred)), 4)})
        log(results[-1])
json.dump(results, open(os.path.join(HERE, 'nhtsa_task_results.json'), 'w'), indent=1)
# paired randomisation between fields: swap the two predictions per record at random
perm_rng = np.random.default_rng(SEED)


def paired(labels, pred_a, pred_b, n=5000):
    d0 = macro_f1(labels, pred_a) - macro_f1(labels, pred_b)
    count = 0
    for _ in range(n):
        swap = perm_rng.random(len(labels)) < 0.5
        if abs(macro_f1(labels, np.where(swap, pred_b, pred_a)) - macro_f1(labels, np.where(swap, pred_a, pred_b))) >= abs(d0) - 1e-12:
            count += 1
    return d0, (count + 1) / (n + 1)


contrasts = []
pairs = [('summary', 'conseq'), ('summary', 'remedy'), ('remedy', 'conseq')]
for arch, seeds in (('tfidf', (0,)), ('bilstm', (0, 1, 2))):
    for field_a, field_b in pairs:
        for seed in seeds:
            preds_a = np.load(os.path.join(HERE, f'nhtsa_preds_{field_a}_{arch}_s{seed}.npz'))
            preds_b = np.load(os.path.join(HERE, f'nhtsa_preds_{field_b}_{arch}_s{seed}.npz'))
            d0, p = paired(preds_a['y'], preds_a['pred'], preds_b['pred'])
            contrasts.append({"contrast": f"{arch} s{seed}: {field_a} vs {field_b}", "delta": round(float(d0), 4), "p": round(float(p), 4)})
            log(contrasts[-1])
n_contrasts = len(contrasts)
for i, contrast in enumerate(sorted(contrasts, key=lambda c: c["p"])):
    contrast["p_holm"] = round(min(1.0, contrast["p"] * (n_contrasts - i)), 4)
json.dump(contrasts, open(os.path.join(HERE, 'nhtsa_task_contrasts.json'), 'w'), indent=1)
log("NHTSA TASK DONE")
