"""Two probes on the trained BiLSTM/GloVe-200 setting: shuffled token order
(train and test on a fixed permutation per record) and a 512-token cap.
Cheap way to ask what the sequence model is actually using -- order, or just
the ability to aggregate. Appends to results/ablation_results.jsonl.
"""
import argparse
import importlib
import os
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["L1_RES"] = os.path.join(HERE, "results_ablate")
sys.path.insert(0, HERE)
l1 = importlib.import_module("l1_train")

import time
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["shuffle", "cap512"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.mode == "cap512":
        l1.MAX_LEN = 512
    recs, test_acns = l1.load_data()
    if args.mode == "shuffle":
        for r in recs:
            rng = np.random.default_rng(l1.GLOBAL_DATA_SEED
                                        + zlib.crc32(str(r["acn"]).encode()))
            rng.shuffle(r["toks"])
    kp = ("smoke_" if args.smoke else "") + args.mode + "_"
    if args.smoke:
        rng = np.random.default_rng(0)
        recs = [recs[i] for i in rng.choice(len(recs), 3000, replace=False)]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    idx, matrix = l1.build_matrix("glove200", recs)
    X, y = l1.encode(recs, idx)
    is_test = np.array([r["acn"] in test_acns for r in recs])
    Xtr_all, ytr_all = X[~is_test], y[~is_test]
    Xte, yte = X[is_test], y[is_test]
    done = l1.done_keys()
    print(f"{args.mode}: {len(recs)} recs, matrix {matrix.shape}, device {device}")

    for seed in (0, 1, 2):
        key = f"{kp}final_glove200_bilstm_s{seed}"
        if key in done:
            print(key, "done, skipping")
            continue
        Xtr, Xva, ytr, yva = train_test_split(
            Xtr_all, ytr_all, test_size=0.05, stratify=ytr_all,
            random_state=l1.GLOBAL_DATA_SEED + seed)
        t0 = time.time()
        _, val_f1, pte = l1.run_one(Xtr, ytr, Xva, yva, Xte, matrix, "bilstm",
                                    seed=100 + seed, device=device,
                                    smoke=args.smoke)
        f1 = f1_score(yte, (pte >= 0.5).astype(int), average="macro")
        np.savez(os.path.join(l1.RES_DIR, f"{kp}preds_glove200_bilstm_s{seed}.npz"),
                 probs=pte, y=yte)
        l1.emit({"key": key, "mode": "final", "emb": "glove200",
                 "arch": "bilstm", "seed": seed, "ablation": args.mode,
                 "test_macro_f1": round(float(f1), 4),
                 "val_f1": round(float(val_f1), 4),
                 "secs": round(time.time() - t0)})
        print(key, round(float(f1), 4))
        if args.smoke:
            break


if __name__ == "__main__":
    main()
