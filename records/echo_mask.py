"""Trains one BiLSTM per record type on the record-trained word2vec and
scores the held-out set twice: as is, and with the tokens that also
appear in the ASRS category labels replaced by the OOV index in both
record types. Reads records_task.jsonl.gz and w2v_<record>_200d.kv;
writes echo_mask.json.
Run: python3 echo_mask.py"""
import gzip
import json
import re
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors

ECHO = {'acft', 'aircraft', 'equip', 'equipment', 'prob', 'problem', 'malfunction',
        'malfunctioning', 'critical', 'nmac', 'atc', 'wx', 'weather'}
TOKEN_RE = re.compile(r"[a-z][a-z0-9/-]+")
SEED = 20260802
PAD, OOV = 0, 1
CAPS = {'narr': 256, 'syn': 64}
records = [json.loads(line) for line in gzip.open('records_task.jsonl.gz', 'rt')]
device = 'mps' if torch.backends.mps.is_available() else 'cpu'


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


out = {}
for record in ('syn', 'narr'):
    cap = CAPS[record]
    token_lists = [TOKEN_RE.findall(r[record].lower())[:cap] for r in records]
    vocab = set(t for ts in token_lists for t in ts)
    keyed = KeyedVectors.load(f"w2v_{record}_200d.kv", mmap='r')
    rng = np.random.default_rng(0)
    index = {}
    vectors = [np.zeros(200, np.float32), rng.normal(0, 0.1, 200).astype(np.float32)]
    for word in sorted(vocab):
        if word in keyed:
            index[word] = len(vectors)
            vectors.append(keyed[word].astype(np.float32))
    matrix = np.stack(vectors)

    def encode(lists, mask=False):
        X = np.zeros((len(lists), cap), np.int32)
        for i, tokens in enumerate(lists):
            for j, token in enumerate(tokens):
                X[i, j] = OOV if (mask and token in ECHO) else index.get(token, OOV)
        return X

    y = np.array([int(r['label']) for r in records], np.int64)
    is_test = np.array([r['split'] == 'test' for r in records])
    X = encode(token_lists)
    Xm = encode(token_lists, mask=True)
    echo_rate = np.mean([any(t in ECHO for t in ts) for ts, m in zip(token_lists, is_test) if m])
    Xtr, Xva, ytr, yva = train_test_split(X[~is_test], y[~is_test], test_size=0.05, stratify=y[~is_test], random_state=SEED)
    torch.manual_seed(100)
    model = RNN(matrix).to(device)
    optimiser = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    Xtr_t = torch.from_numpy(Xtr).long()
    ytr_t = torch.from_numpy(ytr)
    best, best_state, patience = -1, None, 0
    for epoch in range(15):
        model.train()
        perm = torch.randperm(len(Xtr_t), generator=torch.Generator().manual_seed(100000 + epoch))
        for start in range(0, len(perm), 128):
            batch = perm[start:start + 128]
            optimiser.zero_grad()
            loss_fn(model(Xtr_t[batch].to(device)), ytr_t[batch].to(device)).backward()
            optimiser.step()
        f1 = f1_score(yva, (predict(model, Xva) >= 0.5).astype(int), average='macro')
        if f1 > best + 1e-4:
            best, patience = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 2:
                break
    model.load_state_dict(best_state)
    f1_plain = f1_score(y[is_test], (predict(model, X[is_test]) >= 0.5).astype(int), average='macro')
    f1_masked = f1_score(y[is_test], (predict(model, Xm[is_test]) >= 0.5).astype(int), average='macro')
    out[record] = {"plain": round(float(f1_plain), 4), "echo_masked": round(float(f1_masked), 4),
                   "cost": round(float(f1_plain - f1_masked), 4), "echo_hit_rate": round(float(echo_rate), 3)}
    print(record, out[record])
json.dump(out, open('echo_mask.json', 'w'), indent=1)
