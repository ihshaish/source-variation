"""Runs the transformer models on the Aircraft task, frozen or fine-tuned.

frozen: the transformer only extracts features (attention-masked mean of the
last hidden layer, cached to disk), and a one-hidden-layer head is trained
on them with the same 10-fold and three-seed setup as train.py. This is the
contextual counterpart of the mean-pooling MLP: same head, same setup, only
the representation differs.
finetune: DistilBERT is fine-tuned end to end, three seeded final models
scored on the held-out set. There is no cross-validation in this mode.

Reads task_aircraft.jsonl.gz and split.json from ASRS_DATA and appends to
asrs_results.jsonl in ASRS_RES, under the keys ctxfrozen_* and ctxft_*. Both
variables default to the data and results folders next to this script.

python3 train_transformers.py --model distilbert|safeaero --mode frozen|finetune [--smoke]
"""
import argparse
import gzip
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("ASRS_DATA", os.path.join(HERE, "data"))
RES = os.environ.get("ASRS_RES", os.path.join(HERE, "results"))
SEED = 20260802
MODELS = {"distilbert": "distilbert-base-uncased",
          "safeaero": "NASA-AIML/MIKA_SafeAeroBERT"}
DEV = ("mps" if torch.backends.mps.is_available()
       else "cuda" if torch.cuda.is_available() else "cpu")


def macro_f1(y, yhat):
    out = 0.0
    for label in (0, 1):
        tp = ((yhat == label) & (y == label)).sum()
        fp = ((yhat == label) & (y != label)).sum()
        fn = ((yhat != label) & (y == label)).sum()
        denom = 2 * tp + fp + fn
        out += 2 * tp / denom if denom else 0.0
    return out / 2


def load_task(smoke=False):
    with open(os.path.join(DATA, "split.json")) as f:
        test_ids = set(json.load(f)["test_acns"])
    texts = []
    labels = []
    is_test = []
    with gzip.open(os.path.join(DATA, "task_aircraft.jsonl.gz"), "rt") as f:
        for line in f:
            record = json.loads(line)
            texts.append(record["text"])
            labels.append(record["label"])
            is_test.append(record["acn"] in test_ids)
    if smoke:
        rng = np.random.default_rng(0)
        keep = rng.choice(len(texts), 300, replace=False)
        texts = [texts[i] for i in keep]
        labels = [labels[i] for i in keep]
        is_test = [is_test[i] for i in keep]
    return texts, np.array(labels), np.array(is_test)


def extract_features(tag, texts, smoke):
    cache = os.path.join(DATA, f"ctxfeat_{tag}{'_smoke' if smoke else ''}.npz")
    if os.path.exists(cache):
        return np.load(cache)["X"]
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODELS[tag])
    model = AutoModel.from_pretrained(MODELS[tag]).to(DEV).eval()
    features = []
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(texts), 64):
            encoded = tokenizer(texts[start:start + 64], truncation=True, max_length=256,
                                padding=True, return_tensors="pt").to(DEV)
            out = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(2).float()
            features.append(((out * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().numpy())
            if start % 6400 == 0:
                print(f"  extract {start}/{len(texts)} ({time.time()-t0:.0f}s)", flush=True)
    X = np.concatenate(features).astype(np.float32)
    np.savez_compressed(cache, X=X)
    return X


class Head(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, 128)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        return self.fc2(self.drop(torch.relu(self.fc1(x))))


def train_head(Xtr, ytr, Xva, yva, Xev, seed, smoke):
    torch.manual_seed(seed)
    model = Head(Xtr.shape[1]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(Xtr).to(DEV)
    yt = torch.from_numpy(ytr).to(DEV)
    best = -1.0
    state = None
    patience = 0
    for epoch in range(1 if smoke else 30):
        model.train()
        perm = torch.randperm(len(Xt), generator=torch.Generator().manual_seed(seed * 999 + epoch)).to(DEV)
        for start in range(0, len(perm), 256):
            batch = perm[start:start + 256]
            opt.zero_grad()
            loss_fn(model(Xt[batch]), yt[batch]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_probs = model(torch.from_numpy(Xva).to(DEV)).softmax(1)[:, 1].cpu().numpy()
        f1 = macro_f1(yva, (val_probs >= 0.5).astype(int))
        if f1 > best + 1e-4:
            best = f1
            patience = 0
            state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 3:
                break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        eval_probs = model(torch.from_numpy(Xev).to(DEV)).softmax(1)[:, 1].cpu().numpy()
    return best, eval_probs


def emit(result):
    with open(os.path.join(RES, "asrs_results.jsonl"), "a") as f:
        f.write(json.dumps(result) + "\n")
    print(json.dumps(result), flush=True)


def done():
    path = os.path.join(RES, "asrs_results.jsonl")
    if not os.path.exists(path):
        return set()
    return {json.loads(line)["key"] for line in open(path)}


def run_frozen(tag, smoke):
    texts, y, is_test = load_task(smoke)
    X = extract_features(tag, texts, smoke)
    Xtr_all = X[~is_test]
    ytr_all = y[~is_test]
    Xte = X[is_test]
    yte = y[is_test]
    key_prefix = "smoke_" if smoke else ""
    done_keys = done()
    skf = StratifiedKFold(10, shuffle=True, random_state=SEED)
    for fold, (train_index, eval_index) in enumerate(skf.split(Xtr_all, ytr_all)):
        key = f"{key_prefix}cv_ctxfrozen-{tag}_mlp_f{fold}"
        if key in done_keys:
            continue
        Xtr, Xva, ytr, yva = train_test_split(Xtr_all[train_index], ytr_all[train_index],
                                              test_size=0.05, stratify=ytr_all[train_index],
                                              random_state=SEED + fold)
        _, eval_probs = train_head(Xtr, ytr, Xva, yva, Xtr_all[eval_index], fold, smoke)
        emit({"key": key, "mode": "cv", "emb": f"ctxfrozen-{tag}", "arch": "mlp",
              "fold": fold, "macro_f1": round(float(macro_f1(ytr_all[eval_index], (eval_probs >= 0.5).astype(int))), 4)})
        if smoke:
            break
    for seed in (0, 1, 2):
        key = f"{key_prefix}final_ctxfrozen-{tag}_mlp_s{seed}"
        if key in done_keys:
            continue
        Xtr, Xva, ytr, yva = train_test_split(Xtr_all, ytr_all, test_size=0.05,
                                              stratify=ytr_all, random_state=SEED + seed)
        _, test_probs = train_head(Xtr, ytr, Xva, yva, Xte, 100 + seed, smoke)
        np.savez(os.path.join(RES, f"{key_prefix}preds_ctxfrozen-{tag}_mlp_s{seed}.npz"),
                 probs=test_probs, y=yte)
        emit({"key": key, "mode": "final", "emb": f"ctxfrozen-{tag}", "arch": "mlp",
              "seed": seed, "test_macro_f1": round(float(macro_f1(yte, (test_probs >= 0.5).astype(int))), 4)})
        if smoke:
            break


def run_finetune(tag, smoke):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    texts, y, is_test = load_task(smoke)
    train_index = np.where(~is_test)[0]
    test_index = np.where(is_test)[0]
    tokenizer = AutoTokenizer.from_pretrained(MODELS[tag])
    key_prefix = "smoke_" if smoke else ""
    done_keys = done()
    for seed in (0, 1, 2):
        key = f"{key_prefix}final_ctxft-{tag}_s{seed}"
        if key in done_keys:
            continue
        torch.manual_seed(seed)
        rng = np.random.default_rng(SEED + seed)
        val_set = set(rng.choice(train_index, max(1, int(0.05 * len(train_index))), replace=False))
        train_rows = [i for i in train_index if i not in val_set]
        val_rows = sorted(val_set)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODELS[tag], num_labels=2).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
        best = -1.0
        state = None
        t0 = time.time()

        def batches(rows, batch_size, shuffle_seed=None):
            rows = list(rows)
            if shuffle_seed is not None:
                np.random.default_rng(shuffle_seed).shuffle(rows)
            for start in range(0, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                encoded = tokenizer([texts[i] for i in batch], truncation=True, max_length=256,
                                    padding=True, return_tensors="pt").to(DEV)
                yield batch, encoded

        def score(rows):
            model.eval()
            probs = []
            with torch.no_grad():
                for batch, encoded in batches(rows, 64):
                    probs.append(model(**encoded).logits.softmax(1)[:, 1].cpu().numpy())
            return np.concatenate(probs)

        for epoch in range(1 if smoke else 3):
            model.train()
            for step, (batch, encoded) in enumerate(batches(train_rows, 16, shuffle_seed=seed * 7 + epoch)):
                loss = nn.functional.cross_entropy(model(**encoded).logits,
                                                   torch.tensor([y[i] for i in batch]).to(DEV))
                opt.zero_grad()
                loss.backward()
                opt.step()
                if step % 500 == 0:
                    print(f"  ft {tag} s{seed} e{epoch} step {step} ({time.time()-t0:.0f}s)", flush=True)
            f1 = macro_f1(y[val_rows], (score(val_rows) >= 0.5).astype(int))
            print(f"  ft {tag} s{seed} epoch {epoch} val {f1:.4f}", flush=True)
            if f1 > best + 1e-4:
                best = f1
                state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                break
        model.load_state_dict(state)
        model.to(DEV)
        test_probs = score(test_index)
        yte = y[test_index]
        np.savez(os.path.join(RES, f"{key_prefix}preds_ctxft-{tag}_s{seed}.npz"), probs=test_probs, y=yte)
        emit({"key": key, "mode": "final", "emb": f"ctxft-{tag}", "arch": "finetune",
              "seed": seed, "val_f1": round(float(best), 4),
              "test_macro_f1": round(float(macro_f1(yte, (test_probs >= 0.5).astype(int))), 4)})
        if smoke:
            break


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["distilbert", "safeaero"])
    ap.add_argument("--mode", required=True, choices=["frozen", "finetune"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.mode == "frozen":
        run_frozen(args.model, args.smoke)
    else:
        run_finetune(args.model, args.smoke)
