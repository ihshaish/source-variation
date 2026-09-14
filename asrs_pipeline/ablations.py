"""Two ablations of the BiLSTM with GloVe-200.

shuffle permutes the tokens of every record with a fixed per-record seed,
for training and test alike, so the model sees the same words with no
order. cap512 raises the token cap from 256 to 512. Each mode trains the
three final seeds and scores them on the held-out set, reusing the code in
train.py. Results go to asrs_results.jsonl in results_ablate next to this
script, with the mode in the key.

python3 ablations.py --mode shuffle|cap512 [--smoke]
"""
import argparse
import importlib
import os
import sys
import time
import zlib

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["ASRS_RES"] = os.path.join(HERE, "results_ablate")
sys.path.insert(0, HERE)
train = importlib.import_module("train")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["shuffle", "cap512"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.mode == "cap512":
        train.MAX_LEN = 512
    records, test_acns = train.load_data()
    if args.mode == "shuffle":
        for record in records:
            rng = np.random.default_rng(train.GLOBAL_DATA_SEED
                                        + zlib.crc32(str(record["acn"]).encode()))
            rng.shuffle(record["toks"])
    key_prefix = ("smoke_" if args.smoke else "") + args.mode + "_"
    if args.smoke:
        rng = np.random.default_rng(0)
        records = [records[i] for i in rng.choice(len(records), 3000, replace=False)]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    index, matrix = train.build_matrix("glove200", records)
    X, y = train.encode(records, index)
    is_test = np.array([record["acn"] in test_acns for record in records])
    Xtr_all = X[~is_test]
    ytr_all = y[~is_test]
    Xte = X[is_test]
    yte = y[is_test]
    done = train.done_keys()
    print(f"{args.mode}: {len(records)} recs, matrix {matrix.shape}, device {device}")

    for seed in (0, 1, 2):
        key = f"{key_prefix}final_glove200_bilstm_s{seed}"
        if key in done:
            print(key, "done, skipping")
            continue
        Xtr, Xva, ytr, yva = train_test_split(
            Xtr_all, ytr_all, test_size=0.05, stratify=ytr_all,
            random_state=train.GLOBAL_DATA_SEED + seed)
        t0 = time.time()
        _, val_f1, test_probs = train.run_one(Xtr, ytr, Xva, yva, Xte, matrix, "bilstm",
                                              seed=100 + seed, device=device,
                                              smoke=args.smoke)
        f1 = f1_score(yte, (test_probs >= 0.5).astype(int), average="macro")
        np.savez(os.path.join(train.RES_DIR, f"{key_prefix}preds_glove200_bilstm_s{seed}.npz"),
                 probs=test_probs, y=yte)
        train.emit({"key": key, "mode": "final", "emb": "glove200",
                    "arch": "bilstm", "seed": seed, "ablation": args.mode,
                    "test_macro_f1": round(float(f1), 4),
                    "val_f1": round(float(val_f1), 4),
                    "secs": round(time.time() - t0)})
        print(key, round(float(f1), 4))
        if args.smoke:
            break


if __name__ == "__main__":
    main()
