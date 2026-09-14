"""Trains the 2x2 record matrix on the dual-report subset, balanced: both
models train only on the dual-subset training records, one on the primary
narrative and one on the second reporter's narrative, three seeds each,
and each is scored on both narratives of the dual held-out records and on
the length-matched subset. Reads records_task.jsonl.gz and
w2v_narr_200d.kv; writes matrix_2x2.json.
Run: python3 matrix_2x2.py"""
import gzip
import json
import re
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors

TOKEN_RE = re.compile(r"[a-z][a-z0-9/-]+")
SEED = 20260802
PAD, OOV = 0, 1
CAP = 256
records = [json.loads(line) for line in gzip.open('records_task.jsonl.gz', 'rt')]
dual = [r for r in records if len(r['r2']) > 40]
print("dual records:", len(dual), "train", sum(1 for r in dual if r['split'] == 'train'))
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
keyed = KeyedVectors.load("w2v_narr_200d.kv", mmap='r')


def tokenise(text):
    return TOKEN_RE.findall(text.lower())[:CAP]


for r in dual:
    r['t1'] = tokenise(r['narr'])
    r['t2'] = tokenise(r['r2'])
vocab = set(t for r in dual for t in r['t1'] + r['t2'])
rng = np.random.default_rng(0)
index = {}
vectors = [np.zeros(200, np.float32), rng.normal(0, 0.1, 200).astype(np.float32)]
for word in sorted(vocab):
    if word in keyed:
        index[word] = len(vectors)
        vectors.append(keyed[word].astype(np.float32))
matrix = np.stack(vectors)


def encode(key):
    X = np.zeros((len(dual), CAP), np.int32)
    for i, r in enumerate(dual):
        for j, token in enumerate(r[key]):
            X[i, j] = index.get(token, OOV)
    return X


X1, X2 = encode('t1'), encode('t2')
y = np.array([int(r['label']) for r in dual], np.int64)
is_test = np.array([r['split'] == 'test' for r in dual])
len_r1 = np.array([len(r['t1']) for r in dual])
len_r2 = np.array([len(r['t2']) for r in dual])
# length-matched subset: second narrative at least 70% as long as the first
matched = is_test & (len_r2 >= 0.7 * len_r1)


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


def macro_f1(y, probs):
    return f1_score(y, (probs >= 0.5).astype(int), average='macro')


results = []
for train_record, Xtr_full in (('R1', X1), ('R2', X2)):
    for seed in (0, 1, 2):
        Xtr, Xva, ytr, yva = train_test_split(Xtr_full[~is_test], y[~is_test], test_size=0.08, stratify=y[~is_test], random_state=SEED + seed)
        torch.manual_seed(500 + seed)
        model = RNN(matrix).to(device)
        optimiser = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        Xtr_t = torch.from_numpy(Xtr).long()
        ytr_t = torch.from_numpy(ytr)
        best, best_state, patience = -1, None, 0
        for epoch in range(20):
            model.train()
            perm = torch.randperm(len(Xtr_t), generator=torch.Generator().manual_seed(seed * 777 + epoch))
            for start in range(0, len(perm), 64):
                batch = perm[start:start + 64]
                optimiser.zero_grad()
                loss_fn(model(Xtr_t[batch].to(device)), ytr_t[batch].to(device)).backward()
                optimiser.step()
            f1 = macro_f1(yva, predict(model, Xva))
            if f1 > best + 1e-4:
                best, patience = f1, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= 3:
                    break
        model.load_state_dict(best_state)
        row = {"train": train_record, "seed": seed,
               "test_R1": round(float(macro_f1(y[is_test], predict(model, X1[is_test]))), 4),
               "test_R2": round(float(macro_f1(y[is_test], predict(model, X2[is_test]))), 4),
               "match_R1": round(float(macro_f1(y[matched], predict(model, X1[matched]))), 4),
               "match_R2": round(float(macro_f1(y[matched], predict(model, X2[matched]))), 4)}
        results.append(row)
        print(json.dumps(row), flush=True)
json.dump(results, open('matrix_2x2.json', 'w'), indent=1)
