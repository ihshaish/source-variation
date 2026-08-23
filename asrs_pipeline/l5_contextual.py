"""L5 — contextual-representation probes and the seeded transformer fine-tune
(co-author suggestion + plan item L4), NASA Aircraft task, v2 corpus.

Two modes, keeping the paper's attribution design:
- frozen: the transformer is a feature extractor only (attention-masked mean of
  the last hidden layer); a one-hidden-layer head trains on the cached features
  under the full 10-fold and 3-seed setup. This is the contextual analogue of
  the mean-pooling probe: same head, same setup, only the representation
  changes.
- finetune: end-to-end fine-tuning (DistilBERT), three seeded finals on the
  held-out set. No CV here (compute); the held-out bootstrap stage applies.

python3 l5_contextual.py --model distilbert|safeaero --mode frozen|finetune [--smoke]
Results append to results_v2/l1_results.jsonl (keys ctxfrozen_*/ctxft_*).
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
DATA = os.path.join(HERE, "l1_data_v2")
RES = os.path.join(HERE, "results_v2")
SEED = 20260802
MODELS = {"distilbert": "distilbert-base-uncased",
          "safeaero": "NASA-AIML/MIKA_SafeAeroBERT"}
DEV = ("mps" if torch.backends.mps.is_available()
       else "cuda" if torch.cuda.is_available() else "cpu")


def macro_f1(y, yhat):
    out = 0.0
    for c in (0, 1):
        tp = ((yhat == c) & (y == c)).sum()
        fp = ((yhat == c) & (y != c)).sum()
        fn = ((yhat != c) & (y == c)).sum()
        d = 2 * tp + fp + fn
        out += 2 * tp / d if d else 0.0
    return out / 2


def load_task(smoke=False):
    test_ids = set(json.load(open(os.path.join(DATA, "split.json")))["test_acns"])
    texts, labels, is_test = [], [], []
    with gzip.open(os.path.join(DATA, "task_aircraft.jsonl.gz"), "rt") as f:
        for line in f:
            r = json.loads(line)
            texts.append(r["text"])
            labels.append(r["label"])
            is_test.append(r["acn"] in test_ids)
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
    tok = AutoTokenizer.from_pretrained(MODELS[tag])
    mdl = AutoModel.from_pretrained(MODELS[tag]).to(DEV).eval()
    feats = []
    t0 = time.time()
    with torch.no_grad():
        for b in range(0, len(texts), 64):
            enc = tok(texts[b:b + 64], truncation=True, max_length=256,
                      padding=True, return_tensors="pt").to(DEV)
            out = mdl(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(2).float()
            feats.append(((out * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy())
            if b % 6400 == 0:
                print(f"  extract {b}/{len(texts)} ({time.time()-t0:.0f}s)", flush=True)
    X = np.concatenate(feats).astype(np.float32)
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
    lossf = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(Xtr).to(DEV)
    yt = torch.from_numpy(ytr).to(DEV)
    best, state, patience = -1.0, None, 0
    for epoch in range(1 if smoke else 30):
        model.train()
        perm = torch.randperm(len(Xt), generator=torch.Generator().manual_seed(seed * 999 + epoch)).to(DEV)
        for b in range(0, len(perm), 256):
            sel = perm[b:b + 256]
            opt.zero_grad()
            lossf(model(Xt[sel]), yt[sel]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(torch.from_numpy(Xva).to(DEV)).softmax(1)[:, 1].cpu().numpy()
        f1 = macro_f1(yva, (pv >= .5).astype(int))
        if f1 > best + 1e-4:
            best, patience = f1, 0
            state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 3:
                break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        pe = model(torch.from_numpy(Xev).to(DEV)).softmax(1)[:, 1].cpu().numpy()
    return best, pe


def emit(rec):
    with open(os.path.join(RES, "l1_results.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def done():
    p = os.path.join(RES, "l1_results.jsonl")
    return {json.loads(l)["key"] for l in open(p)} if os.path.exists(p) else set()


def run_frozen(tag, smoke):
    texts, y, is_test = load_task(smoke)
    X = extract_features(tag, texts, smoke)
    Xtr_all, ytr_all = X[~is_test], y[~is_test]
    Xte, yte = X[is_test], y[is_test]
    kp = "smoke_" if smoke else ""
    dn = done()
    skf = StratifiedKFold(10, shuffle=True, random_state=SEED)
    for fold, (itr, iev) in enumerate(skf.split(Xtr_all, ytr_all)):
        key = f"{kp}cv_ctxfrozen-{tag}_mlp_f{fold}"
        if key in dn:
            continue
        Xtr, Xva, ytr, yva = train_test_split(Xtr_all[itr], ytr_all[itr],
                                              test_size=0.05, stratify=ytr_all[itr],
                                              random_state=SEED + fold)
        _, pe = train_head(Xtr, ytr, Xva, yva, Xtr_all[iev], fold, smoke)
        emit({"key": key, "mode": "cv", "emb": f"ctxfrozen-{tag}", "arch": "mlp",
              "fold": fold, "macro_f1": round(float(macro_f1(ytr_all[iev], (pe >= .5).astype(int))), 4)})
        if smoke:
            break
    for seed in (0, 1, 2):
        key = f"{kp}final_ctxfrozen-{tag}_mlp_s{seed}"
        if key in dn:
            continue
        Xtr, Xva, ytr, yva = train_test_split(Xtr_all, ytr_all, test_size=0.05,
                                              stratify=ytr_all, random_state=SEED + seed)
        _, pe = train_head(Xtr, ytr, Xva, yva, Xte, 100 + seed, smoke)
        np.savez(os.path.join(RES, f"{kp}preds_ctxfrozen-{tag}_mlp_s{seed}.npz"),
                 probs=pe, y=yte)
        emit({"key": key, "mode": "final", "emb": f"ctxfrozen-{tag}", "arch": "mlp",
              "seed": seed, "test_macro_f1": round(float(macro_f1(yte, (pe >= .5).astype(int))), 4)})
        if smoke:
            break


def run_finetune(tag, smoke):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    texts, y, is_test = load_task(smoke)
    tr_idx = np.where(~is_test)[0]
    te_idx = np.where(is_test)[0]
    tok = AutoTokenizer.from_pretrained(MODELS[tag])
    kp = "smoke_" if smoke else ""
    dn = done()
    for seed in (0, 1, 2):
        key = f"{kp}final_ctxft-{tag}_s{seed}"
        if key in dn:
            continue
        torch.manual_seed(seed)
        rng = np.random.default_rng(SEED + seed)
        val = set(rng.choice(tr_idx, max(1, int(0.05 * len(tr_idx))), replace=False))
        tr = [i for i in tr_idx if i not in val]
        va = sorted(val)
        mdl = AutoModelForSequenceClassification.from_pretrained(
            MODELS[tag], num_labels=2).to(DEV)
        opt = torch.optim.AdamW(mdl.parameters(), lr=2e-5)
        best, state, t0 = -1.0, None, time.time()

        def batches(ix, bs, shuffle_seed=None):
            ix = list(ix)
            if shuffle_seed is not None:
                np.random.default_rng(shuffle_seed).shuffle(ix)
            for b in range(0, len(ix), bs):
                sel = ix[b:b + bs]
                enc = tok([texts[i] for i in sel], truncation=True, max_length=256,
                          padding=True, return_tensors="pt").to(DEV)
                yield sel, enc

        def score(ix):
            mdl.eval()
            ps = []
            with torch.no_grad():
                for sel, enc in batches(ix, 64):
                    ps.append(mdl(**enc).logits.softmax(1)[:, 1].cpu().numpy())
            return np.concatenate(ps)

        for epoch in range(1 if smoke else 3):
            mdl.train()
            for n, (sel, enc) in enumerate(batches(tr, 16, shuffle_seed=seed * 7 + epoch)):
                loss = nn.functional.cross_entropy(mdl(**enc).logits,
                                                   torch.tensor([y[i] for i in sel]).to(DEV))
                opt.zero_grad()
                loss.backward()
                opt.step()
                if n % 500 == 0:
                    print(f"  ft {tag} s{seed} e{epoch} step {n} ({time.time()-t0:.0f}s)", flush=True)
            f1 = macro_f1(y[va], (score(va) >= .5).astype(int))
            print(f"  ft {tag} s{seed} epoch {epoch} val {f1:.4f}", flush=True)
            if f1 > best + 1e-4:
                best = f1
                state = {k: v.cpu().clone() for k, v in mdl.state_dict().items()}
            else:
                break
        mdl.load_state_dict(state)
        mdl.to(DEV)
        pe = score(te_idx)
        yte = y[te_idx]
        np.savez(os.path.join(RES, f"{kp}preds_ctxft-{tag}_s{seed}.npz"), probs=pe, y=yte)
        emit({"key": key, "mode": "final", "emb": f"ctxft-{tag}", "arch": "finetune",
              "seed": seed, "val_f1": round(float(best), 4),
              "test_macro_f1": round(float(macro_f1(yte, (pe >= .5).astype(int))), 4)})
        if smoke:
            break


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["distilbert", "safeaero"])
    ap.add_argument("--mode", required=True, choices=["frozen", "finetune"])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    (run_frozen if a.mode == "frozen" else run_finetune)(a.model, a.smoke)
