"""Recomputes the paired contrasts on each training seed.

paired_stats.py tests the seed-0 final models. This script repeats each
contrast on the seed-0, seed-1 and seed-2 final models and reports the
delta in macro-F1 and the paired randomisation p-value for every seed, so
the reader can see how much the result moves with the training seed.

Reads the preds_*_s?.npz files from ASRS_RES and writes seed_stability.json
there. Run: python3 seed_stability.py
"""
import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("ASRS_RES", os.path.join(HERE, "results"))
B = 10_000
rng = np.random.default_rng(20260802)


def macro_f1(y, yhat):
    out = 0.0
    for label in (0, 1):
        tp = np.count_nonzero((yhat == label) & (y == label))
        fp = np.count_nonzero((yhat == label) & (y != label))
        fn = np.count_nonzero((yhat != label) & (y == label))
        denom = 2 * tp + fp + fn
        out += (2 * tp / denom) if denom else 0.0
    return out / 2


def paired_randomisation(preds_a, preds_b, y):
    observed = abs(macro_f1(y, preds_a) - macro_f1(y, preds_b))
    differs = preds_a != preds_b
    n = len(y)
    count = 0
    for _ in range(B):
        swapped = (rng.random(n) < 0.5) & differs
        a_swapped = np.where(swapped, preds_b, preds_a)
        b_swapped = np.where(swapped, preds_a, preds_b)
        if abs(macro_f1(y, a_swapped) - macro_f1(y, b_swapped)) >= observed - 1e-12:
            count += 1
    return (count + 1) / (B + 1)


preds = {}
for path in glob.glob(os.path.join(RES, "preds_*_s?.npz")):
    match = re.match(r"preds_(.+)_(bilstm|bigru|cnn|meanmlp)_s(\d)\.npz", os.path.basename(path))
    if match:
        saved = np.load(path)
        config = (match.group(1), match.group(2), int(match.group(3)))
        preds[config] = ((saved["probs"] >= 0.5).astype(int), saved["y"])

CONTRASTS = [
    ("B-arch@w2v", ("w2vasrs", "bilstm"), ("w2vasrs", "bigru")),
    ("B-arch@glove50", ("glove50", "bilstm"), ("glove50", "bigru")),
    ("C-subword@bilstm", ("fasttext", "bilstm"), ("w2vasrs", "bilstm")),
    ("C-subword@bigru", ("fasttext", "bigru"), ("w2vasrs", "bigru")),
    ("A-indomain@bigru", ("glove200", "bigru"), ("w2vasrs", "bigru")),
    ("A-indomain@bilstm", ("glove200", "bilstm"), ("w2vasrs", "bilstm")),
    ("D-seq-mlp", ("glove200", "bilstm"), ("glove200", "meanmlp")),
    ("D-seq-cnn", ("glove200", "bilstm"), ("glove200", "cnn")),
]
out = {}
for name, config_a, config_b in CONTRASTS:
    rows = []
    for seed in (0, 1, 2):
        if (*config_a, seed) not in preds or (*config_b, seed) not in preds:
            continue
        preds_a, y = preds[(*config_a, seed)]
        preds_b, _ = preds[(*config_b, seed)]
        delta = macro_f1(y, preds_a) - macro_f1(y, preds_b)
        p = paired_randomisation(preds_a, preds_b, y)
        rows.append({"seed": seed, "delta": round(float(delta), 4), "p": round(p, 4)})
    out[name] = rows
    print(name, rows)
with open(os.path.join(RES, "seed_stability.json"), "w") as f:
    json.dump(out, f, indent=1)
