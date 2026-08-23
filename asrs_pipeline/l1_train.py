"""One embedding x one architecture on the Aircraft task.

Frozen embedding (OOV goes to a shared fallback vector), BiLSTM or BiGRU,
10-fold CV inside the training partition, then three final models (seeds 0-2)
scored once on the held-out 20%. Per-record predictions are saved because
l1_stats.py does all the statistics later -- this script just trains.

Run --smoke first: two minutes, and it tells you whether your data and GloVe
paths are right before you commit a GPU evening to the full queue.

python3 l1_train.py --emb glove200 --arch bilstm [--smoke]
Results append to results/l1_results.jsonl, keyed, so reruns are safe.
"""
import argparse
import gzip
import json
import os
import re
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
EMB_DIR = os.path.join(HERE, "..", "embeddings")
RES_DIR = os.environ.get("L1_RES", os.path.join(HERE, "results"))
DATA = os.environ.get("L1_DATA", os.path.join(HERE, "l1_data"))
os.makedirs(RES_DIR, exist_ok=True)

TOKEN_RE = re.compile(r"[a-z][a-z0-9/-]+")
GLOBAL_DATA_SEED = 20260802
MAX_LEN = 256
PAD, OOV = 0, 1
GLOVE_FILES = {"glove50": ("glove.6B.50d.txt", 50), "glove100": ("glove.6B.100d.txt", 100),
               "glove200": ("glove.6B.200d.txt", 200), "glove300": ("glove.6B.300d.txt", 300)}


def load_data():
    with open(os.path.join(DATA, "split.json")) as f:
        test_acns = set(json.load(f)["test_acns"])
    recs = []
    with gzip.open(os.path.join(DATA, "task_aircraft.jsonl.gz"), "rt") as f:
        for line in f:
            r = json.loads(line)
            r["toks"] = TOKEN_RE.findall(r.pop("text").lower())[:MAX_LEN]
            recs.append(r)
    return recs, test_acns


def build_matrix(emb_name, recs):
    """index map + frozen matrix. Row 0 pad (zeros), row 1 shared OOV fallback."""
    dataset_vocab = set()
    for r in recs:
        dataset_vocab.update(r["toks"])
    rng = np.random.default_rng(0)
    if emb_name in GLOVE_FILES:
        fn, dim = GLOVE_FILES[emb_name]
        idx, vecs = {}, [np.zeros(dim, np.float32), rng.normal(0, 0.1, dim).astype(np.float32)]
        with open(os.path.join(EMB_DIR, fn), encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                w = parts[0]
                if w in dataset_vocab:
                    idx[w] = len(vecs)
                    vecs.append(np.asarray(parts[1:], np.float32))
        return idx, np.stack(vecs)
    if emb_name == "fasttext":
        from gensim.models import FastText
        ft = FastText.load(os.path.join(DATA, "fasttext_asrs_train_200d.bin"), mmap="r")
        dim = ft.wv.vector_size
        idx, vecs = {}, [np.zeros(dim, np.float32), rng.normal(0, 0.1, dim).astype(np.float32)]
        for w in sorted(dataset_vocab):
            idx[w] = len(vecs)
            vecs.append(ft.wv[w].astype(np.float32))  # subword composition, incl. unseen
        return idx, np.stack(vecs)
    if emb_name == "w2vasrs":
        # control for fastText: same corpus/params, no subwords -> OOV falls back
        from gensim.models import Word2Vec
        w2v = Word2Vec.load(os.path.join(DATA, "w2v_asrs_train_200d.bin"), mmap="r")
        dim = w2v.wv.vector_size
        idx, vecs = {}, [np.zeros(dim, np.float32), rng.normal(0, 0.1, dim).astype(np.float32)]
        for w in sorted(dataset_vocab):
            if w in w2v.wv:
                idx[w] = len(vecs)
                vecs.append(w2v.wv[w].astype(np.float32))
        return idx, np.stack(vecs)
    raise ValueError(emb_name)


def encode(recs, idx):
    X = np.zeros((len(recs), MAX_LEN), np.int32)
    for i, r in enumerate(recs):
        for j, t in enumerate(r["toks"]):
            X[i, j] = idx.get(t, OOV)
    y = np.array([r["label"] for r in recs], np.int64)
    return X, y


class RNN(nn.Module):
    def __init__(self, matrix, arch, hidden=64, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(torch.from_numpy(matrix),
                                                freeze=True, padding_idx=PAD)
        cls = nn.LSTM if arch == "bilstm" else nn.GRU
        self.rnn = cls(matrix.shape[1], hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden, 2)

    def forward(self, x):
        e = self.emb(x)
        out = self.rnn(e)
        h = out[1][0] if isinstance(self.rnn, nn.LSTM) else out[1]
        h = torch.cat([h[0], h[1]], dim=1)          # terminal fwd+bwd hidden
        return self.fc(self.drop(h))


class TextCNN(nn.Module):
    """Non-recurrent sequence probe: conv widths 3/4/5, 100 filters each,
    global max pool (Kim 2014)."""
    def __init__(self, matrix, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(torch.from_numpy(matrix),
                                                freeze=True, padding_idx=PAD)
        d = matrix.shape[1]
        self.convs = nn.ModuleList([nn.Conv1d(d, 100, k, padding=k // 2) for k in (3, 4, 5)])
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(300, 2)

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)
        h = torch.cat([torch.relu(c(e)).amax(dim=2) for c in self.convs], dim=1)
        return self.fc(self.drop(h))


class MeanMLP(nn.Module):
    """Order-free probe: mask-aware mean of frozen embeddings, one hidden layer.
    Separates sequence modelling from representation."""
    def __init__(self, matrix, hidden=128, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(torch.from_numpy(matrix),
                                                freeze=True, padding_idx=PAD)
        self.fc1 = nn.Linear(matrix.shape[1], hidden)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 2)

    def forward(self, x):
        e = self.emb(x)
        mask = (x != PAD).unsqueeze(2).float()
        m = (e * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.fc2(self.drop(torch.relu(self.fc1(m))))


def build_model(matrix, arch):
    if arch in ("bilstm", "bigru"):
        return RNN(matrix, arch)
    if arch == "cnn":
        return TextCNN(matrix)
    if arch == "meanmlp":
        return MeanMLP(matrix)
    raise ValueError(arch)


def run_one(Xtr, ytr, Xva, yva, Xev, matrix, arch, seed, device, smoke=False):
    torch.manual_seed(seed)
    model = build_model(matrix, arch).to(device)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    Xtr_t = torch.from_numpy(Xtr).long()
    ytr_t = torch.from_numpy(ytr)
    best_f1, best_state, patience = -1.0, None, 0
    max_epochs = 1 if smoke else 15
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t), generator=torch.Generator().manual_seed(seed * 1000 + epoch))
        for b in range(0, len(perm), 128):
            sel = perm[b:b + 128]
            opt.zero_grad()
            loss = lossf(model(Xtr_t[sel].to(device)), ytr_t[sel].to(device))
            loss.backward()
            opt.step()
        pv = predict(model, Xva, device)
        f1 = f1_score(yva, pv, average="macro")
        if f1 > best_f1 + 1e-4:
            best_f1, patience = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 2:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_f1, predict(model, Xev, device, probs=True)


def predict(model, X, device, probs=False):
    model.eval()
    outs = []
    with torch.no_grad():
        for b in range(0, len(X), 256):
            logits = model(torch.from_numpy(X[b:b + 256]).long().to(device))
            outs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    p = np.concatenate(outs)
    return p if probs else (p >= 0.5).astype(np.int64)


def done_keys():
    path = os.path.join(RES_DIR, "l1_results.jsonl")
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {json.loads(line)["key"] for line in f if line.strip()}


def emit(rec):
    with open(os.path.join(RES_DIR, "l1_results.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--arch", required=True, choices=["bilstm", "bigru", "cnn", "meanmlp"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    # cross-lane guard: if another process is already running this exact config,
    # yield (parallel lanes own disjoint configs; the sequential chain skips them)
    import subprocess as sp
    probe = sp.run(["pgrep", "-f", f"l1_train.py --emb {args.emb} --arch {args.arch}"],
                   capture_output=True, text=True)
    others = [int(p) for p in probe.stdout.split() if p.strip().isdigit() and int(p) != os.getpid()]
    if others:
        print(f"config {args.emb}/{args.arch} already running in pid {others}; yielding")
        return
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    recs, test_acns = load_data()
    kp = "smoke_" if args.smoke else ""      # smoke keys never collide with real ones
    if args.smoke:
        rng = np.random.default_rng(0)
        recs = [recs[i] for i in rng.choice(len(recs), 3000, replace=False)]
    idx, matrix = build_matrix(args.emb, recs)
    print(f"{args.emb}/{args.arch}: vocab-in-emb {len(idx)}, matrix {matrix.shape}, device {device}")
    X, y = encode(recs, idx)
    is_test = np.array([r["acn"] in test_acns for r in recs])
    Xtr_all, ytr_all = X[~is_test], y[~is_test]
    Xte, yte = X[is_test], y[is_test]
    done = done_keys()

    # ---- 10-fold CV in the training partition
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=GLOBAL_DATA_SEED)
    for fold, (itr, iev) in enumerate(skf.split(Xtr_all, ytr_all)):
        key = f"{kp}cv_{args.emb}_{args.arch}_f{fold}"
        if key in done:
            continue
        Xf, yf = Xtr_all[itr], ytr_all[itr]
        Xtr, Xva, ytr, yva = train_test_split(Xf, yf, test_size=0.05, stratify=yf,
                                              random_state=GLOBAL_DATA_SEED + fold)
        t0 = time.time()
        _, val_f1, pev = run_one(Xtr, ytr, Xva, yva, Xtr_all[iev], matrix, args.arch,
                                 seed=fold, device=device, smoke=args.smoke)
        f1 = f1_score(ytr_all[iev], (pev >= 0.5).astype(int), average="macro")
        emit({"key": key, "mode": "cv", "emb": args.emb, "arch": args.arch, "fold": fold,
              "macro_f1": round(float(f1), 4), "val_f1": round(float(val_f1), 4),
              "secs": round(time.time() - t0)})
        if args.smoke:
            break

    # ---- final models on full training partition, scored on held-out test
    for seed in (0, 1, 2):
        key = f"{kp}final_{args.emb}_{args.arch}_s{seed}"
        if key in done:
            continue
        Xtr, Xva, ytr, yva = train_test_split(Xtr_all, ytr_all, test_size=0.05,
                                              stratify=ytr_all, random_state=GLOBAL_DATA_SEED + seed)
        t0 = time.time()
        _, val_f1, pte = run_one(Xtr, ytr, Xva, yva, Xte, matrix, args.arch,
                                 seed=100 + seed, device=device, smoke=args.smoke)
        f1 = f1_score(yte, (pte >= 0.5).astype(int), average="macro")
        np.savez(os.path.join(RES_DIR, f"{kp}preds_{args.emb}_{args.arch}_s{seed}.npz"),
                 probs=pte, y=yte)
        emit({"key": key, "mode": "final", "emb": args.emb, "arch": args.arch, "seed": seed,
              "test_macro_f1": round(float(f1), 4), "val_f1": round(float(val_f1), 4),
              "secs": round(time.time() - t0)})
        if args.smoke:
            break


if __name__ == "__main__":
    main()
