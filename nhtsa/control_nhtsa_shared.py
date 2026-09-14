"""Trains the BiLSTM of nhtsa_task.py with one word2vec model shared across
fields. The task build and split are the same as in nhtsa_task.py, but a
single word2vec is trained once on the training text of all three fields
together, then frozen and reused for summary, consequence and remedy, so only
the field changes between runs. Three seeds per field, then paired
randomisation tests between fields with Holm correction over the nine
contrasts. Writes one nhtsa_shared_preds_*.npz per run and
control_nhtsa_shared.json. Run: python3 control_nhtsa_shared.py"""
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
from collections import Counter


def top_component(component):
    return (component or '').split(':')[0].split(',')[0].strip()


class_counts = Counter(top_component(record['Component']) for record in campaigns)
classes = sorted([name for name, count in class_counts.items() if count >= 300 and name])
label_index = {name: i for i, name in enumerate(classes)}
data = [record for record in campaigns if top_component(record['Component']) in label_index]
rng = np.random.default_rng(SEED)
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
for record in data:
    record['split'] = 'test' if record['NHTSACampaignNumber'] in test_campaigns else 'train'
y = np.array([label_index[top_component(record['Component'])] for record in data])
is_test = np.array([record['split'] == 'test' for record in data])
log("task:", len(data), "test:", int(is_test.sum()), "classes:", len(classes))

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def macro_f1(labels, pred):
    return f1_score(labels, pred, average='macro')


FIELDS = {'summary': 'Summary', 'conseq': 'Consequence', 'remedy': 'Remedy'}
CAP = 96
PAD, OOV = 0, 1

from gensim.models import Word2Vec
tokens_by_field = {field_key: [TOKEN_RE.findall((record[field] or '').lower())[:CAP] for record in data] for field_key, field in FIELDS.items()}
shared_sentences = [t for field_key in FIELDS for t, m in zip(tokens_by_field[field_key], ~is_test) if m and t]
w2v = Word2Vec(vector_size=200, window=5, min_count=3, sg=1, epochs=10, workers=8, seed=SEED)
w2v.build_vocab(shared_sentences)
w2v.train(corpus_iterable=shared_sentences, total_examples=len(shared_sentences), epochs=10)
vocab = set(t for field_key in FIELDS for ts in tokens_by_field[field_key] for t in ts)
oov_rng = np.random.default_rng(0)
word_index, vectors = {}, [np.zeros(200, np.float32), oov_rng.normal(0, 0.1, 200).astype(np.float32)]
for word in sorted(vocab):
    if word in w2v.wv:
        word_index[word] = len(vectors)
        vectors.append(w2v.wv[word].astype(np.float32))
matrix = np.stack(vectors)
log("shared w2v vocab:", len(word_index))

import torch
import torch.nn as nn
device = 'mps' if torch.backends.mps.is_available() else 'cpu'


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


results = []
predictions = {}
for field_key in FIELDS:
    token_lists = tokens_by_field[field_key]
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
        predictions[(field_key, seed)] = pred
        np.savez(os.path.join(HERE, f'nhtsa_shared_preds_{field_key}_bilstm_s{seed}.npz'), pred=pred, y=y[is_test])
        results.append({"key": f"nhtsaS_{field_key}_bilstm_s{seed}", "f1": round(float(macro_f1(y[is_test], pred)), 4)})
        log(results[-1])

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


labels = y[is_test]
contrasts = []
for seed in (0, 1, 2):
    for field_a, field_b in [('summary', 'conseq'), ('summary', 'remedy'), ('remedy', 'conseq')]:
        delta, p = paired(labels, predictions[(field_a, seed)], predictions[(field_b, seed)])
        contrasts.append({"contrast": f"shared bilstm s{seed}: {field_a} vs {field_b}", "delta": round(float(delta), 4), "p": round(float(p), 4)})
        log(contrasts[-1])
contrasts_sorted = sorted(contrasts, key=lambda c: c["p"])
n_contrasts = len(contrasts_sorted)
for i, contrast in enumerate(contrasts_sorted):
    contrast["p_holm"] = round(min(1.0, max((n_contrasts - j) * contrasts_sorted[j]["p"] for j in range(i + 1))), 4)
json.dump({"results": results, "contrasts": contrasts}, open(os.path.join(HERE, 'control_nhtsa_shared.json'), 'w'), indent=1)
print("NHTSA_SHARED DONE", flush=True)
