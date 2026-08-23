"""Seed-stability check for the primary contrasts: recompute each pre-specified
paired contrast on the seed-1 and seed-2 final models (seed-0 is the reported
stage) and report the range of deltas and p-values across seeds. Answers the
'fixed-model tests ignore training-seed variation' objection empirically."""
import glob, json, os, re
import numpy as np

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_v2")
B = 10_000
rng = np.random.default_rng(20260802)

def macro_f1(y, yhat):
    out = 0.0
    for c in (0, 1):
        tp = np.count_nonzero((yhat == c) & (y == c))
        fp = np.count_nonzero((yhat == c) & (y != c))
        fn = np.count_nonzero((yhat != c) & (y == c))
        d = 2 * tp + fp + fn
        out += (2 * tp / d) if d else 0.0
    return out / 2

def rand_p(a, b, y):
    obs = abs(macro_f1(y, a) - macro_f1(y, b))
    diff = a != b
    n = len(y); count = 0
    for _ in range(B):
        s = (rng.random(n) < 0.5) & diff
        aa = np.where(s, b, a); bb = np.where(s, a, b)
        if abs(macro_f1(y, aa) - macro_f1(y, bb)) >= obs - 1e-12:
            count += 1
    return (count + 1) / (B + 1)

P = {}
for path in glob.glob(os.path.join(RES, "preds_*_s?.npz")):
    m = re.match(r"preds_(.+)_(bilstm|bigru|cnn|meanmlp)_s(\d)\.npz", os.path.basename(path))
    if m:
        d = np.load(path)
        P[(m.group(1), m.group(2), int(m.group(3)))] = ((d["probs"] >= 0.5).astype(int), d["y"])

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
for name, c1, c2 in CONTRASTS:
    rows = []
    for s in (0, 1, 2):
        if (*c1, s) not in P or (*c2, s) not in P:
            continue
        a, y = P[(*c1, s)]; b, _ = P[(*c2, s)]
        d = macro_f1(y, a) - macro_f1(y, b)
        p = rand_p(a, b, y)
        rows.append({"seed": s, "delta": round(float(d), 4), "p": round(p, 4)})
    out[name] = rows
    print(name, rows)
json.dump(out, open(os.path.join(RES, "seed_stability.json"), "w"), indent=1)
