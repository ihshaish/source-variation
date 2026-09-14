"""Trains the mean-pool MLP, the order-free probe, on both record types
over the record-trained word2vec, three seeds each, and saves the
held-out predictions. Reads records_task.jsonl.gz and
w2v_<record>_200d.kv; writes preds_<record>_meanmlp_s<seed>.npz.
Run: python3 records_meanpool.py"""
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
CAPS = {'narr': 256, 'syn': 64}
records = [json.loads(line) for line in gzip.open('records_task.jsonl.gz', 'rt')]
device = 'mps' if torch.backends.mps.is_available() else 'cpu'


class MeanMLP(nn.Module):
    def __init__(self, matrix, hidden=128):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(torch.from_numpy(matrix), freeze=True, padding_idx=PAD)
        self.fc1 = nn.Linear(matrix.shape[1], hidden)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden, 2)

    def forward(self, x):
        embedded = self.emb(x)
        mask = (x != PAD).unsqueeze(2).float()
        pooled = (embedded * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.fc2(self.drop(torch.relu(self.fc1(pooled))))


def predict(model, X):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(X), 512):
            outputs.append(torch.softmax(model(torch.from_numpy(X[start:start + 512]).long().to(device)), 1)[:, 1].cpu().numpy())
    return np.concatenate(outputs)


def macro_f1(y, probs):
    return f1_score(y, (probs >= 0.5).astype(int), average='macro')


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
    X = np.zeros((len(token_lists), cap), np.int32)
    for i, tokens in enumerate(token_lists):
        for j, token in enumerate(tokens):
            X[i, j] = index.get(token, OOV)
    y = np.array([int(r['label']) for r in records], np.int64)
    is_test = np.array([r['split'] == 'test' for r in records])
    for seed in (0, 1, 2):
        Xtr, Xva, ytr, yva = train_test_split(X[~is_test], y[~is_test], test_size=0.05, stratify=y[~is_test], random_state=SEED + seed)
        torch.manual_seed(300 + seed)
        model = MeanMLP(matrix).to(device)
        optimiser = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        Xtr_t = torch.from_numpy(Xtr).long()
        ytr_t = torch.from_numpy(ytr)
        best, best_state, patience = -1, None, 0
        for epoch in range(15):
            model.train()
            perm = torch.randperm(len(Xtr_t), generator=torch.Generator().manual_seed(seed * 991 + epoch))
            for start in range(0, len(perm), 128):
                batch = perm[start:start + 128]
                optimiser.zero_grad()
                loss_fn(model(Xtr_t[batch].to(device)), ytr_t[batch].to(device)).backward()
                optimiser.step()
            f1 = macro_f1(yva, predict(model, Xva))
            if f1 > best + 1e-4:
                best, patience = f1, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= 2:
                    break
        model.load_state_dict(best_state)
        probs = predict(model, X[is_test])
        np.savez(f'preds_{record}_meanmlp_s{seed}.npz', probs=probs, y=y[is_test])
        print(json.dumps({"view": record, "arch": "meanmlp", "seed": seed, "test_macro_f1": round(float(macro_f1(y[is_test], probs)), 4)}), flush=True)
print("MEANPOOL DONE", flush=True)
