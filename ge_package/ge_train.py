"""Trains one configuration on the GE task.

python ge_train.py --field repair --emb avi2vec --arch bilstm
        [--split random] [--mask-file masks/vdelta.txt] [--smoke]

field: customer | technician | repair | sequential
emb:   glove50 | glove100 | glove200 | glove300 | avi2vec | fasttext_ge | w2v_ge
arch:  bilstm | bigru | cnn | meanmlp
mask-file: newline-separated tokens replaced by the OOV vector before
  encoding; the mask file name becomes part of the result key.

Runs ten-fold cross-validation in the training partition and three final
seeds on the held-out set. Appends to ge_results.jsonl under GE_RES, one
line per key and never twice for the same key, and saves the per-record
prediction arrays as preds_<key>_s<seed>.npz (integers only, no text).
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", required=True,
                        choices=["customer", "technician", "repair", "sequential"])
    parser.add_argument("--emb", required=True)
    parser.add_argument("--arch", required=True,
                        choices=["bilstm", "bigru", "cnn", "meanmlp"])
    parser.add_argument("--split", default="random")
    parser.add_argument("--mask-file", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    mask = None
    mask_tag = ""
    if args.mask_file:
        mask = {line.strip() for line in open(args.mask_file) if line.strip()}
        mask_tag = "_mask-" + os.path.splitext(os.path.basename(args.mask_file))[0]
    key_prefix = "smoke_" if args.smoke else ""
    base_key = f"{args.split}_{args.field}_{args.emb}_{args.arch}{mask_tag}"
    results_path = os.path.join(RES, "ge_results.jsonl")

    records = load_records()
    if args.smoke:
        rng = np.random.default_rng(0)
        chosen = rng.choice(len(records), min(300, len(records)), replace=False)
        records = [records[i] for i in chosen]
    test_ids = set(json.load(open(os.path.join(DATA, "splits.json")))[args.split])
    vocab = set()
    for record in records:
        vocab.update(record_tokens(record, args.field, mask))
    index, matrix = build_matrix(args.emb, vocab)
    X, y = encode(records, args.field, index, mask)
    is_test = np.array([record["id"] in test_ids for record in records])
    Xtr_all, ytr_all, Xte, yte = X[~is_test], y[~is_test], X[is_test], y[is_test]
    print(f"{base_key}: train {len(ytr_all)}, test {len(yte)}, vocab-in-emb {len(index)}")
    done = done_keys(results_path)

    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=GLOBAL_SEED)
    for fold, (train_index, eval_index) in enumerate(folds.split(Xtr_all, ytr_all)):
        key = f"{key_prefix}cv_{base_key}_f{fold}"
        if key in done:
            continue
        Xtr, Xva, ytr, yva = train_test_split(
            Xtr_all[train_index], ytr_all[train_index], test_size=0.05,
            stratify=ytr_all[train_index], random_state=GLOBAL_SEED + fold)
        _, val_f1, eval_predictions = train_eval(Xtr, ytr, Xva, yva, Xtr_all[eval_index], matrix,
                                                 args.arch, seed=fold, smoke=args.smoke)
        emit(results_path, {"key": key, "mode": "cv", "macro_f1":
                            round(float(macro_f1(ytr_all[eval_index], eval_predictions)), 4),
                            "val_f1": round(float(val_f1), 4)})
        if args.smoke:
            break

    for seed in (0, 1, 2):
        key = f"{key_prefix}final_{base_key}_s{seed}"
        if key in done:
            continue
        Xtr, Xva, ytr, yva = train_test_split(
            Xtr_all, ytr_all, test_size=0.05, stratify=ytr_all,
            random_state=GLOBAL_SEED + seed)
        _, val_f1, test_predictions = train_eval(Xtr, ytr, Xva, yva, Xte, matrix, args.arch,
                                                 seed=100 + seed, smoke=args.smoke)
        np.savez(os.path.join(RES, f"preds_{key_prefix}{base_key}_s{seed}.npz"),
                 yhat=test_predictions, y=yte)
        emit(results_path, {"key": key, "mode": "final", "test_macro_f1":
                            round(float(macro_f1(yte, test_predictions)), 4),
                            "val_f1": round(float(val_f1), 4)})
        if args.smoke:
            break


if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)
    main()
