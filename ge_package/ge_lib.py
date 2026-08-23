"""Shared core for the GE-side experiment package (Paper A).

Runs entirely inside GE: reads the records CSV and embedding files locally,
writes only aggregate results (JSONL/JSON/TeX) and per-record prediction
arrays (integers, no text). See README.md for the input schema and for what
may leave GE.

Design mirrors the public-corpus reimplementation (paper_a_nasa) so the two
sides of the paper share one setup: frozen embeddings, OOV -> shared
fallback vector, max length per field, batch 128, Adam 1e-3, early stopping
patience 2 / max 15 epochs, 10-fold CV in the 80% train partition + 3 final
seeds scored on the held-out 20%.

Boundary tokens (sequential field): three reserved indices with FIXED, seeded
random vectors (distinct, non-trainable) - documented so the provenance
encoding is explicit.
"""
import csv
import gzip
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("GE_DATA", os.path.join(HERE, "ge_data"))
RES = os.environ.get("GE_RES", os.path.join(HERE, "results"))
TOKEN_RE = re.compile(r"[a-z][a-z0-9/-]+")
GLOBAL_SEED = 20260802
MAX_LEN = {"customer": 64, "technician": 128, "repair": 128, "sequential": 256}
PAD, OOV, B_CUST, B_TECH, B_REP = 0, 1, 2, 3, 4
N_RESERVED = 5
FIELDS = ("customer", "technician", "repair")
csv.field_size_limit(10_000_000)


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def load_records():
    """ge_records.jsonl.gz written by ge_build.py."""
    recs = []
    with gzip.open(os.path.join(DATA, "ge_records.jsonl.gz"), "rt") as f:
        for line in f:
            recs.append(json.loads(line))
    return recs


def record_tokens(rec, field, mask=None):
    if field == "sequential":
        toks = (["<B_CUST>"] + tokenize(rec["customer"]) +
                ["<B_TECH>"] + tokenize(rec["technician"]) +
                ["<B_REP>"] + tokenize(rec["repair"]))
    else:
        toks = tokenize(rec[field])
    if mask:
        toks = ["<MASKED>" if t in mask else t for t in toks]
    return toks[:MAX_LEN[field if field in MAX_LEN else "sequential"]]


def build_matrix(emb_name, vocab_needed, emb_dir=None):
    """Frozen matrix. Rows: 0 pad, 1 OOV, 2-4 boundary (fixed seeded vectors)."""
    emb_dir = emb_dir or os.environ.get("EMB_DIR", os.path.join(HERE, "..", "embeddings"))
    rng = np.random.default_rng(0)

    def reserved(dim):
        rows = [np.zeros(dim, np.float32)]
        for _ in range(N_RESERVED - 1):        # OOV + 3 boundary tokens, distinct
            rows.append(rng.normal(0, 0.1, dim).astype(np.float32))
        return rows

    glove = {"glove50": ("glove.6B.50d.txt", 50), "glove100": ("glove.6B.100d.txt", 100),
             "glove200": ("glove.6B.200d.txt", 200), "glove300": ("glove.6B.300d.txt", 300)}
    if emb_name in glove:
        fn, dim = glove[emb_name]
        vecs = reserved(dim)
        idx = {}
        with open(os.path.join(emb_dir, fn), encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                if parts[0] in vocab_needed:
                    idx[parts[0]] = len(vecs)
                    vecs.append(np.asarray(parts[1:], np.float32))
        return idx, np.stack(vecs)
    if emb_name == "avi2vec":
        from gensim.models import KeyedVectors
        path = os.path.join(DATA, "avi2vec.kv")
        try:
            kv = KeyedVectors.load(path, mmap="r")
        except Exception:
            kv = KeyedVectors.load_word2vec_format(path, binary=path.endswith(".bin"))
        dim = kv.vector_size
        vecs = reserved(dim)
        idx = {}
        for w in vocab_needed:
            if w in kv:
                idx[w] = len(vecs)
                vecs.append(np.asarray(kv[w], np.float32))
        return idx, np.stack(vecs)
    if emb_name in ("fasttext_ge", "w2v_ge"):
        from gensim.models import FastText, Word2Vec
        cls = FastText if emb_name == "fasttext_ge" else Word2Vec
        m = cls.load(os.path.join(DATA, f"{emb_name}_200d.bin"), mmap="r")
        dim = m.wv.vector_size
        vecs = reserved(dim)
        idx = {}
        for w in sorted(vocab_needed):
            if emb_name == "fasttext_ge" or w in m.wv:
                idx[w] = len(vecs)
                vecs.append(m.wv[w].astype(np.float32))
        return idx, np.stack(vecs)
    raise ValueError(emb_name)


def encode(recs, field, idx, mask=None):
    L = MAX_LEN[field if field in MAX_LEN else "sequential"]
    special = {"<B_CUST>": B_CUST, "<B_TECH>": B_TECH, "<B_REP>": B_REP, "<MASKED>": OOV}
    X = np.zeros((len(recs), L), np.int32)
    for i, r in enumerate(recs):
        for j, t in enumerate(record_tokens(r, field, mask)):
            X[i, j] = special.get(t) or idx.get(t, OOV)
    y = np.array([r["label"] for r in recs], np.int64)
    return X, y


def macro_f1(y, yhat, n_classes=4):
    out = 0.0
    for c in range(n_classes):
        tp = np.count_nonzero((yhat == c) & (y == c))
        fp = np.count_nonzero((yhat == c) & (y != c))
        fn = np.count_nonzero((yhat != c) & (y == c))
        d = 2 * tp + fp + fn
        out += (2 * tp / d) if d else 0.0
    return out / n_classes


def make_model(matrix, arch, n_classes=4):
    import torch
    import torch.nn as nn

    class RNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding.from_pretrained(torch.from_numpy(matrix),
                                                    freeze=True, padding_idx=PAD)
            cls = nn.LSTM if arch == "bilstm" else nn.GRU
            self.rnn = cls(matrix.shape[1], 64, batch_first=True, bidirectional=True)
            self.drop = nn.Dropout(0.3)
            self.fc = nn.Linear(128, n_classes)

        def forward(self, x):
            out = self.rnn(self.emb(x))
            h = out[1][0] if arch == "bilstm" else out[1]
            return self.fc(self.drop(torch.cat([h[0], h[1]], dim=1)))

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding.from_pretrained(torch.from_numpy(matrix),
                                                    freeze=True, padding_idx=PAD)
            self.convs = nn.ModuleList(
                [nn.Conv1d(matrix.shape[1], 100, k, padding=k // 2) for k in (3, 4, 5)])
            self.drop = nn.Dropout(0.3)
            self.fc = nn.Linear(300, n_classes)

        def forward(self, x):
            e = self.emb(x).transpose(1, 2)
            h = torch.cat([torch.relu(c(e)).amax(dim=2) for c in self.convs], dim=1)
            return self.fc(self.drop(h))

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding.from_pretrained(torch.from_numpy(matrix),
                                                    freeze=True, padding_idx=PAD)
            self.fc1 = nn.Linear(matrix.shape[1], 128)
            self.drop = nn.Dropout(0.3)
            self.fc2 = nn.Linear(128, n_classes)

        def forward(self, x):
            import torch as T
            e = self.emb(x)
            m = (x != PAD).unsqueeze(2).float()
            v = (e * m).sum(1) / m.sum(1).clamp(min=1.0)
            return self.fc2(self.drop(T.relu(self.fc1(v))))

    return {"bilstm": RNN, "bigru": RNN, "cnn": CNN, "meanmlp": MLP}[arch]()


def train_eval(Xtr, ytr, Xva, yva, Xev, matrix, arch, seed, n_classes=4, smoke=False):
    import torch
    import torch.nn as nn
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model = make_model(matrix, arch, n_classes).to(device)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    Xt, yt = torch.from_numpy(Xtr).long(), torch.from_numpy(ytr)
    best, state, patience = -1.0, None, 0
    for epoch in range(1 if smoke else 15):
        model.train()
        perm = torch.randperm(len(Xt), generator=torch.Generator().manual_seed(seed * 1000 + epoch))
        for b in range(0, len(perm), 128):
            sel = perm[b:b + 128]
            opt.zero_grad()
            lossf(model(Xt[sel].to(device)), yt[sel].to(device)).backward()
            opt.step()
        f1 = macro_f1(yva, predict(model, Xva, device), n_classes)
        if f1 > best + 1e-4:
            best, patience = f1, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 2:
                break
    if state:
        model.load_state_dict(state)
    return model, best, predict(model, Xev, device)


def predict(model, X, device):
    import torch
    model.eval()
    outs = []
    with torch.no_grad():
        for b in range(0, len(X), 256):
            logits = model(torch.from_numpy(X[b:b + 256]).long().to(device))
            outs.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(outs)


def done_keys(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {json.loads(l)["key"] for l in f if l.strip()}


def emit(path, rec):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
