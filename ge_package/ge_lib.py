"""Shared code for the scripts in this folder: tokenising, the frozen
embedding matrix, sequence encoding, the four model architectures, training
with early stopping, macro-F1 and the result-file helpers. Everything runs
inside GE; the record file and the embedding files are read locally, and
only aggregate results and integer prediction arrays are written. The setup
is the same as on the public corpora: frozen embeddings, one shared vector
for out-of-vocabulary tokens, a maximum length per field, batch 128, Adam at
1e-3, early stopping with patience 2 over at most 15 epochs, ten-fold
cross-validation inside the 80% training partition and three final seeds
scored on the held-out 20%.
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
# reserved rows of the embedding matrix: padding, out-of-vocabulary, and the
# three field-boundary tokens used by the sequential field
PAD, OOV, B_CUST, B_TECH, B_REP = 0, 1, 2, 3, 4
N_RESERVED = 5
FIELDS = ("customer", "technician", "repair")
csv.field_size_limit(10_000_000)


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def load_records():
    """Reads ge_records.jsonl.gz, written by ge_build.py."""
    records = []
    with gzip.open(os.path.join(DATA, "ge_records.jsonl.gz"), "rt") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def record_tokens(record, field, mask=None):
    if field == "sequential":
        tokens = (["<B_CUST>"] + tokenize(record["customer"]) +
                  ["<B_TECH>"] + tokenize(record["technician"]) +
                  ["<B_REP>"] + tokenize(record["repair"]))
    else:
        tokens = tokenize(record[field])
    if mask:
        tokens = ["<MASKED>" if token in mask else token for token in tokens]
    return tokens[:MAX_LEN[field if field in MAX_LEN else "sequential"]]


def build_matrix(emb_name, vocab_needed, emb_dir=None):
    """Frozen embedding matrix. Row 0 is padding (zeros); rows 1 to 4 are the
    OOV vector and the three boundary vectors, drawn once from a seeded
    generator so they are fixed and distinct."""
    emb_dir = emb_dir or os.environ.get("EMB_DIR", os.path.join(HERE, "..", "embeddings"))
    rng = np.random.default_rng(0)

    def reserved(dim):
        rows = [np.zeros(dim, np.float32)]
        for _ in range(N_RESERVED - 1):
            rows.append(rng.normal(0, 0.1, dim).astype(np.float32))
        return rows

    glove = {"glove50": ("glove.6B.50d.txt", 50), "glove100": ("glove.6B.100d.txt", 100),
             "glove200": ("glove.6B.200d.txt", 200), "glove300": ("glove.6B.300d.txt", 300)}
    if emb_name in glove:
        file_name, dim = glove[emb_name]
        vectors = reserved(dim)
        index = {}
        with open(os.path.join(emb_dir, file_name), encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                if parts[0] in vocab_needed:
                    index[parts[0]] = len(vectors)
                    vectors.append(np.asarray(parts[1:], np.float32))
        return index, np.stack(vectors)
    if emb_name == "avi2vec":
        from gensim.models import KeyedVectors
        path = os.path.join(DATA, "avi2vec.kv")
        try:
            keyed = KeyedVectors.load(path, mmap="r")
        except Exception:
            keyed = KeyedVectors.load_word2vec_format(path, binary=path.endswith(".bin"))
        dim = keyed.vector_size
        vectors = reserved(dim)
        index = {}
        for word in vocab_needed:
            if word in keyed:
                index[word] = len(vectors)
                vectors.append(np.asarray(keyed[word], np.float32))
        return index, np.stack(vectors)
    if emb_name in ("fasttext_ge", "w2v_ge"):
        from gensim.models import FastText, Word2Vec
        model_class = FastText if emb_name == "fasttext_ge" else Word2Vec
        model = model_class.load(os.path.join(DATA, f"{emb_name}_200d.bin"), mmap="r")
        dim = model.wv.vector_size
        vectors = reserved(dim)
        index = {}
        for word in sorted(vocab_needed):
            if emb_name == "fasttext_ge" or word in model.wv:
                index[word] = len(vectors)
                vectors.append(model.wv[word].astype(np.float32))
        return index, np.stack(vectors)
    raise ValueError(emb_name)


def encode(records, field, index, mask=None):
    max_len = MAX_LEN[field if field in MAX_LEN else "sequential"]
    special = {"<B_CUST>": B_CUST, "<B_TECH>": B_TECH, "<B_REP>": B_REP, "<MASKED>": OOV}
    X = np.zeros((len(records), max_len), np.int32)
    for i, record in enumerate(records):
        for j, token in enumerate(record_tokens(record, field, mask)):
            X[i, j] = special.get(token) or index.get(token, OOV)
    y = np.array([record["label"] for record in records], np.int64)
    return X, y


def macro_f1(y, yhat, n_classes=4):
    total = 0.0
    for label in range(n_classes):
        tp = np.count_nonzero((yhat == label) & (y == label))
        fp = np.count_nonzero((yhat == label) & (y != label))
        fn = np.count_nonzero((yhat != label) & (y == label))
        denominator = 2 * tp + fp + fn
        total += (2 * tp / denominator) if denominator else 0.0
    return total / n_classes


def make_model(matrix, arch, n_classes=4):
    import torch
    import torch.nn as nn

    class RNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding.from_pretrained(torch.from_numpy(matrix),
                                                    freeze=True, padding_idx=PAD)
            rnn_class = nn.LSTM if arch == "bilstm" else nn.GRU
            self.rnn = rnn_class(matrix.shape[1], 64, batch_first=True, bidirectional=True)
            self.drop = nn.Dropout(0.3)
            self.fc = nn.Linear(128, n_classes)

        def forward(self, x):
            out = self.rnn(self.emb(x))
            hidden = out[1][0] if arch == "bilstm" else out[1]
            return self.fc(self.drop(torch.cat([hidden[0], hidden[1]], dim=1)))

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
            embedded = self.emb(x).transpose(1, 2)
            hidden = torch.cat([torch.relu(conv(embedded)).amax(dim=2) for conv in self.convs],
                               dim=1)
            return self.fc(self.drop(hidden))

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding.from_pretrained(torch.from_numpy(matrix),
                                                    freeze=True, padding_idx=PAD)
            self.fc1 = nn.Linear(matrix.shape[1], 128)
            self.drop = nn.Dropout(0.3)
            self.fc2 = nn.Linear(128, n_classes)

        def forward(self, x):
            embedded = self.emb(x)
            not_pad = (x != PAD).unsqueeze(2).float()
            pooled = (embedded * not_pad).sum(1) / not_pad.sum(1).clamp(min=1.0)
            return self.fc2(self.drop(torch.relu(self.fc1(pooled))))

    return {"bilstm": RNN, "bigru": RNN, "cnn": CNN, "meanmlp": MLP}[arch]()


def train_eval(Xtr, ytr, Xva, yva, Xev, matrix, arch, seed, n_classes=4, smoke=False):
    import torch
    import torch.nn as nn
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    torch.manual_seed(seed)
    model = make_model(matrix, arch, n_classes).to(device)
    optimiser = torch.optim.Adam((param for param in model.parameters() if param.requires_grad), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    X_tensor = torch.from_numpy(Xtr).long()
    y_tensor = torch.from_numpy(ytr)
    best = -1.0
    state = None
    patience = 0
    for epoch in range(1 if smoke else 15):
        model.train()
        generator = torch.Generator().manual_seed(seed * 1000 + epoch)
        perm = torch.randperm(len(X_tensor), generator=generator)
        for start in range(0, len(perm), 128):
            batch = perm[start:start + 128]
            optimiser.zero_grad()
            loss_fn(model(X_tensor[batch].to(device)), y_tensor[batch].to(device)).backward()
            optimiser.step()
        f1 = macro_f1(yva, predict(model, Xva, device), n_classes)
        if f1 > best + 1e-4:
            best = f1
            patience = 0
            state = {name: value.detach().cpu().clone()
                     for name, value in model.state_dict().items()}
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
    outputs = []
    with torch.no_grad():
        for start in range(0, len(X), 256):
            logits = model(torch.from_numpy(X[start:start + 256]).long().to(device))
            outputs.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(outputs)


def done_keys(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {json.loads(line)["key"] for line in f if line.strip()}


def emit(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
