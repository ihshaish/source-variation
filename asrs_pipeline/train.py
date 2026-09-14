"""Trains one embedding with one architecture on the Aircraft task.

The embedding is frozen; tokens it does not cover share one fallback vector.
Architectures are BiLSTM, BiGRU, a text CNN and a mean-pooling MLP. Each run
does 10-fold cross-validation inside the training partition, then trains
three final models (seeds 0 to 2) and scores each once on the held-out 20%.
Per-record test probabilities are saved so that paired_stats.py can do the
statistics later.

Reads task_aircraft.jsonl.gz and split.json from ASRS_DATA, GloVe files from
EMB_DIR (default ../embeddings), and the fastText and word2vec models from ASRS_DATA. Appends
one line per fold and per seed to asrs_results.jsonl in ASRS_RES, keyed so
that a rerun skips what is already there.

python3 train.py --emb glove200 --arch bilstm [--smoke]

--smoke trains on 3,000 records for one epoch and one fold; use it to check
that the data and embedding paths are right before a full run.
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
EMB_DIR = os.environ.get("EMB_DIR", os.path.join(HERE, "..", "embeddings"))
RES_DIR = os.environ.get("ASRS_RES", os.path.join(HERE, "results"))
DATA = os.environ.get("ASRS_DATA", os.path.join(HERE, "data"))
os.makedirs(RES_DIR, exist_ok=True)

TOKEN_RE = re.compile(r"[a-z][a-z0-9/-]+")
GLOBAL_DATA_SEED = 20260802
MAX_LEN = 256
PAD = 0
OOV = 1
GLOVE_FILES = {"glove50": ("glove.6B.50d.txt", 50), "glove100": ("glove.6B.100d.txt", 100),
               "glove200": ("glove.6B.200d.txt", 200), "glove300": ("glove.6B.300d.txt", 300)}


def load_data():
    with open(os.path.join(DATA, "split.json")) as f:
        test_acns = set(json.load(f)["test_acns"])
    records = []
    with gzip.open(os.path.join(DATA, "task_aircraft.jsonl.gz"), "rt") as f:
        for line in f:
            record = json.loads(line)
            record["toks"] = TOKEN_RE.findall(record.pop("text").lower())[:MAX_LEN]
            records.append(record)
    return records, test_acns


def build_matrix(emb_name, records):
    """Returns the word index and the frozen matrix. Row 0 is padding (zeros), row 1 is the shared fallback."""
    dataset_vocab = set()
    for record in records:
        dataset_vocab.update(record["toks"])
    rng = np.random.default_rng(0)
    if emb_name in GLOVE_FILES:
        filename, dim = GLOVE_FILES[emb_name]
        index = {}
        vectors = [np.zeros(dim, np.float32), rng.normal(0, 0.1, dim).astype(np.float32)]
        with open(os.path.join(EMB_DIR, filename), encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                word = parts[0]
                if word in dataset_vocab:
                    index[word] = len(vectors)
                    vectors.append(np.asarray(parts[1:], np.float32))
        return index, np.stack(vectors)
    if emb_name == "fasttext":
        from gensim.models import FastText
        fasttext = FastText.load(os.path.join(DATA, "fasttext_asrs_train_200d.bin"), mmap="r")
        dim = fasttext.wv.vector_size
        index = {}
        vectors = [np.zeros(dim, np.float32), rng.normal(0, 0.1, dim).astype(np.float32)]
        # fastText composes a vector from subwords, so every word gets one
        for word in sorted(dataset_vocab):
            index[word] = len(vectors)
            vectors.append(fasttext.wv[word].astype(np.float32))
        return index, np.stack(vectors)
    if emb_name == "w2vasrs":
        from gensim.models import Word2Vec
        w2v = Word2Vec.load(os.path.join(DATA, "w2v_asrs_train_200d.bin"), mmap="r")
        dim = w2v.wv.vector_size
        index = {}
        vectors = [np.zeros(dim, np.float32), rng.normal(0, 0.1, dim).astype(np.float32)]
        for word in sorted(dataset_vocab):
            if word in w2v.wv:
                index[word] = len(vectors)
                vectors.append(w2v.wv[word].astype(np.float32))
        return index, np.stack(vectors)
    raise ValueError(emb_name)


def encode(records, index):
    X = np.zeros((len(records), MAX_LEN), np.int32)
    for i, record in enumerate(records):
        for j, token in enumerate(record["toks"]):
            X[i, j] = index.get(token, OOV)
    y = np.array([record["label"] for record in records], np.int64)
    return X, y


class RNN(nn.Module):
    def __init__(self, matrix, arch, hidden=64, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(torch.from_numpy(matrix),
                                                freeze=True, padding_idx=PAD)
        rnn_class = nn.LSTM if arch == "bilstm" else nn.GRU
        self.rnn = rnn_class(matrix.shape[1], hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden, 2)

    def forward(self, x):
        embedded = self.emb(x)
        out = self.rnn(embedded)
        hidden = out[1][0] if isinstance(self.rnn, nn.LSTM) else out[1]
        # final forward and backward hidden states, side by side
        hidden = torch.cat([hidden[0], hidden[1]], dim=1)
        return self.fc(self.drop(hidden))


class TextCNN(nn.Module):
    """Convolutions of width 3, 4 and 5 with 100 filters each, then global max pooling (Kim 2014)."""
    def __init__(self, matrix, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(torch.from_numpy(matrix),
                                                freeze=True, padding_idx=PAD)
        dim = matrix.shape[1]
        self.convs = nn.ModuleList([nn.Conv1d(dim, 100, k, padding=k // 2) for k in (3, 4, 5)])
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(300, 2)

    def forward(self, x):
        embedded = self.emb(x).transpose(1, 2)
        pooled = torch.cat([torch.relu(conv(embedded)).amax(dim=2) for conv in self.convs], dim=1)
        return self.fc(self.drop(pooled))


class MeanMLP(nn.Module):
    """Mean of the frozen embeddings over the non-padding tokens, then one hidden layer. No word order is used."""
    def __init__(self, matrix, hidden=128, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(torch.from_numpy(matrix),
                                                freeze=True, padding_idx=PAD)
        self.fc1 = nn.Linear(matrix.shape[1], hidden)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 2)

    def forward(self, x):
        embedded = self.emb(x)
        mask = (x != PAD).unsqueeze(2).float()
        mean = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.fc2(self.drop(torch.relu(self.fc1(mean))))


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
    loss_fn = nn.CrossEntropyLoss()
    Xtr_t = torch.from_numpy(Xtr).long()
    ytr_t = torch.from_numpy(ytr)
    best_f1 = -1.0
    best_state = None
    patience = 0
    max_epochs = 1 if smoke else 15
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t), generator=torch.Generator().manual_seed(seed * 1000 + epoch))
        for start in range(0, len(perm), 128):
            batch = perm[start:start + 128]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[batch].to(device)), ytr_t[batch].to(device))
            loss.backward()
            opt.step()
        val_pred = predict(model, Xva, device)
        f1 = f1_score(yva, val_pred, average="macro")
        if f1 > best_f1 + 1e-4:
            best_f1 = f1
            patience = 0
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
        for start in range(0, len(X), 256):
            logits = model(torch.from_numpy(X[start:start + 256]).long().to(device))
            outs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    prob = np.concatenate(outs)
    if probs:
        return prob
    return (prob >= 0.5).astype(np.int64)


def done_keys():
    path = os.path.join(RES_DIR, "asrs_results.jsonl")
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {json.loads(line)["key"] for line in f if line.strip()}


def emit(result):
    with open(os.path.join(RES_DIR, "asrs_results.jsonl"), "a") as f:
        f.write(json.dumps(result) + "\n")
    print(json.dumps(result))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--arch", required=True, choices=["bilstm", "bigru", "cnn", "meanmlp"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    # several queues may run at once on one machine; if another process is
    # already on this exact configuration, leave it to that process
    import subprocess
    probe = subprocess.run(["pgrep", "-f", f"train.py --emb {args.emb} --arch {args.arch}"],
                           capture_output=True, text=True)
    others = [int(p) for p in probe.stdout.split() if p.strip().isdigit() and int(p) != os.getpid()]
    if others:
        print(f"config {args.emb}/{args.arch} already running in pid {others}; yielding")
        return
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    records, test_acns = load_data()
    key_prefix = "smoke_" if args.smoke else ""
    if args.smoke:
        rng = np.random.default_rng(0)
        records = [records[i] for i in rng.choice(len(records), 3000, replace=False)]
    index, matrix = build_matrix(args.emb, records)
    print(f"{args.emb}/{args.arch}: vocab-in-emb {len(index)}, matrix {matrix.shape}, device {device}")
    X, y = encode(records, index)
    is_test = np.array([record["acn"] in test_acns for record in records])
    Xtr_all = X[~is_test]
    ytr_all = y[~is_test]
    Xte = X[is_test]
    yte = y[is_test]
    done = done_keys()

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=GLOBAL_DATA_SEED)
    for fold, (train_index, eval_index) in enumerate(skf.split(Xtr_all, ytr_all)):
        key = f"{key_prefix}cv_{args.emb}_{args.arch}_f{fold}"
        if key in done:
            continue
        Xf = Xtr_all[train_index]
        yf = ytr_all[train_index]
        Xtr, Xva, ytr, yva = train_test_split(Xf, yf, test_size=0.05, stratify=yf,
                                              random_state=GLOBAL_DATA_SEED + fold)
        t0 = time.time()
        _, val_f1, eval_probs = run_one(Xtr, ytr, Xva, yva, Xtr_all[eval_index], matrix, args.arch,
                                        seed=fold, device=device, smoke=args.smoke)
        f1 = f1_score(ytr_all[eval_index], (eval_probs >= 0.5).astype(int), average="macro")
        emit({"key": key, "mode": "cv", "emb": args.emb, "arch": args.arch, "fold": fold,
              "macro_f1": round(float(f1), 4), "val_f1": round(float(val_f1), 4),
              "secs": round(time.time() - t0)})
        if args.smoke:
            break

    for seed in (0, 1, 2):
        key = f"{key_prefix}final_{args.emb}_{args.arch}_s{seed}"
        if key in done:
            continue
        Xtr, Xva, ytr, yva = train_test_split(Xtr_all, ytr_all, test_size=0.05,
                                              stratify=ytr_all, random_state=GLOBAL_DATA_SEED + seed)
        t0 = time.time()
        _, val_f1, test_probs = run_one(Xtr, ytr, Xva, yva, Xte, matrix, args.arch,
                                        seed=100 + seed, device=device, smoke=args.smoke)
        f1 = f1_score(yte, (test_probs >= 0.5).astype(int), average="macro")
        np.savez(os.path.join(RES_DIR, f"{key_prefix}preds_{args.emb}_{args.arch}_s{seed}.npz"),
                 probs=test_probs, y=yte)
        emit({"key": key, "mode": "final", "emb": args.emb, "arch": args.arch, "seed": seed,
              "test_macro_f1": round(float(f1), 4), "val_f1": round(float(val_f1), 4),
              "secs": round(time.time() - t0)})
        if args.smoke:
            break


if __name__ == "__main__":
    main()
