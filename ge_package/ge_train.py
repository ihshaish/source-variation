"""Train one configuration on the GE task.

python ge_train.py --field repair --emb avi2vec --arch bilstm
        [--split random] [--mask-file masks/vdelta.txt] [--smoke]

field: customer | technician | repair | sequential
emb:   glove50 | glove100 | glove200 | glove300 | avi2vec | fasttext_ge | w2v_ge
arch:  bilstm | bigru | cnn | meanmlp
mask-file: newline-separated tokens replaced by the OOV vector before encoding
  (serves the differential-vocabulary masking test, the leakage masking test,
  and the de-identification simulation; the mask name enters the result key).

Runs 10-fold CV in the training partition + 3 final seeds on the held-out set;
appends to results/ge_results.jsonl (idempotent keys); saves per-record
prediction arrays results/preds_<key>_s<seed>.npz (ints only, no text).
"""
import argparse
import json
import os

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from ge_lib import (DATA, GLOBAL_SEED, RES, done_keys, emit, encode,
                    build_matrix, load_records, macro_f1, record_tokens,
                    train_eval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", required=True,
                    choices=["customer", "technician", "repair", "sequential"])
    ap.add_argument("--emb", required=True)
    ap.add_argument("--arch", required=True,
                    choices=["bilstm", "bigru", "cnn", "meanmlp"])
    ap.add_argument("--split", default="random")
    ap.add_argument("--mask-file", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    mask = None
    mtag = ""
    if a.mask_file:
        mask = {l.strip() for l in open(a.mask_file) if l.strip()}
        mtag = "_mask-" + os.path.splitext(os.path.basename(a.mask_file))[0]
    kp = "smoke_" if a.smoke else ""
    base = f"{a.split}_{a.field}_{a.emb}_{a.arch}{mtag}"
    respath = os.path.join(RES, "ge_results.jsonl")

    recs = load_records()
    if a.smoke:
        rng = np.random.default_rng(0)
        recs = [recs[i] for i in rng.choice(len(recs), min(300, len(recs)), replace=False)]
    test_ids = set(json.load(open(os.path.join(DATA, "splits.json")))[a.split])
    vocab = set()
    for r in recs:
        vocab.update(record_tokens(r, a.field, mask))
    idx, matrix = build_matrix(a.emb, vocab)
    X, y = encode(recs, a.field, idx, mask)
    is_test = np.array([r["id"] in test_ids for r in recs])
    Xtr_all, ytr_all, Xte, yte = X[~is_test], y[~is_test], X[is_test], y[is_test]
    print(f"{base}: train {len(ytr_all)}, test {len(yte)}, vocab-in-emb {len(idx)}")
    done = done_keys(respath)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=GLOBAL_SEED)
    for fold, (itr, iev) in enumerate(skf.split(Xtr_all, ytr_all)):
        key = f"{kp}cv_{base}_f{fold}"
        if key in done:
            continue
        Xtr, Xva, ytr, yva = train_test_split(
            Xtr_all[itr], ytr_all[itr], test_size=0.05, stratify=ytr_all[itr],
            random_state=GLOBAL_SEED + fold)
        _, val_f1, pev = train_eval(Xtr, ytr, Xva, yva, Xtr_all[iev], matrix,
                                    a.arch, seed=fold, smoke=a.smoke)
        emit(respath, {"key": key, "mode": "cv", "macro_f1":
                       round(float(macro_f1(ytr_all[iev], pev)), 4),
                       "val_f1": round(float(val_f1), 4)})
        if a.smoke:
            break

    for seed in (0, 1, 2):
        key = f"{kp}final_{base}_s{seed}"
        if key in done:
            continue
        Xtr, Xva, ytr, yva = train_test_split(
            Xtr_all, ytr_all, test_size=0.05, stratify=ytr_all,
            random_state=GLOBAL_SEED + seed)
        _, val_f1, pte = train_eval(Xtr, ytr, Xva, yva, Xte, matrix, a.arch,
                                    seed=100 + seed, smoke=a.smoke)
        np.savez(os.path.join(RES, f"preds_{kp}{base}_s{seed}.npz"), yhat=pte, y=yte)
        emit(respath, {"key": key, "mode": "final", "test_macro_f1":
                       round(float(macro_f1(yte, pte)), 4),
                       "val_f1": round(float(val_f1), 4)})
        if a.smoke:
            break


if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)
    main()
