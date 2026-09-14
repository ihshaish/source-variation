"""Recomputes the interaction D under one shared representation. Trains
the mean-pool MLP on both record types with GloVe-200, the fixed embedding
common to both, with the same setup as records_meanpool.py, then computes
D = (sequence minus pool | syn) minus (sequence minus pool | narr) by
pairing the stored BiLSTM/GloVe-200 predictions with these runs, with a
record bootstrap per seed. Reads records_task.jsonl.gz, glove.6B.200d.txt
from EMB_DIR and preds_<record>_glove200_s<seed>.npz; writes
preds_<record>_meanmlpglove_s<seed>.npz and control_glove_D.json.
Run: python3 control_glove_D.py"""
import gzip
import json
import os
import re
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
GLOVE = os.path.join(os.environ.get('EMB_DIR', '.'), 'glove.6B.200d.txt')
TOKEN_RE = re.compile(r"[a-z][a-z0-9/-]+")
SEED = 20260802
PAD, OOV = 0, 1
CAPS = {'narr': 256, 'syn': 64}
records = [json.loads(line) for line in gzip.open(f'{HERE}/records_task.jsonl.gz', 'rt')]
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
    rng = np.random.default_rng(0)
    index = {}
    vectors = [np.zeros(200, np.float32), rng.normal(0, 0.1, 200).astype(np.float32)]
    for line in open(GLOVE, encoding='utf-8'):
        parts = line.rstrip().split(' ')
        if parts[0] in vocab:
            index[parts[0]] = len(vectors)
            vectors.append(np.asarray(parts[1:], np.float32))
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
        np.savez(f'{HERE}/preds_{record}_meanmlpglove_s{seed}.npz', probs=probs, y=y[is_test])
        print(json.dumps({"view": record, "arch": "meanmlp-glove", "seed": seed,
                          "test_macro_f1": round(float(macro_f1(y[is_test], probs)), 4)}), flush=True)

rng = np.random.default_rng(20260802)


def load(name):
    data = np.load(f'{HERE}/{name}')
    return data['y'], data['probs']


results = []
for seed in (0, 1, 2):
    y, seq_syn = load(f'preds_syn_glove200_s{seed}.npz')
    _, pool_syn = load(f'preds_syn_meanmlpglove_s{seed}.npz')
    _, seq_narr = load(f'preds_narr_glove200_s{seed}.npz')
    _, pool_narr = load(f'preds_narr_meanmlpglove_s{seed}.npz')
    D0 = (macro_f1(y, seq_syn) - macro_f1(y, pool_syn)) - (macro_f1(y, seq_narr) - macro_f1(y, pool_narr))
    n = len(y)
    index = np.arange(n)
    values = []
    for _ in range(5000):
        sample = rng.choice(index, n, replace=True)
        values.append((macro_f1(y[sample], seq_syn[sample]) - macro_f1(y[sample], pool_syn[sample])) - (macro_f1(y[sample], seq_narr[sample]) - macro_f1(y[sample], pool_narr[sample])))
    lo, hi = np.percentile(values, [2.5, 97.5])
    results.append({"seed": seed, "D": round(float(D0), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
                    "excludes_zero": bool(lo > 0 or hi < 0)})
    print(results[-1], flush=True)
json.dump({"interaction_glove": results}, open(f'{HERE}/control_glove_D.json', 'w'), indent=1)
print("GLOVE_D DONE", flush=True)
