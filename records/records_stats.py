"""Paired statistics for the matched-record experiment: a bootstrap
confidence interval per configuration, a paired approximate-randomisation
test for narrative against synopsis on the same records with Holm
correction within the family, and the reporter-1 against reporter-2
paired contrast on the dual-report subset. Reads the preds_*.npz and
dualpreds_*.npz files written by the training scripts and writes
records_stats.json.
Run: python3 records_stats.py"""
import glob
import json
import os
import numpy as np
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(20260802)


def macro_f1(y, probs):
    return f1_score(y, (probs >= 0.5).astype(int), average='macro')


def bootstrap_ci(y, probs, n=10000):
    index = np.arange(len(y))
    values = []
    for _ in range(n):
        sample = rng.choice(index, len(index), replace=True)
        values.append(macro_f1(y[sample], probs[sample]))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def paired(y, probs_a, probs_b, n=10000):
    delta = macro_f1(y, probs_a) - macro_f1(y, probs_b)
    count = 0
    for _ in range(n):
        swap = rng.random(len(y)) < 0.5
        swapped_a = np.where(swap, probs_b, probs_a)
        swapped_b = np.where(swap, probs_a, probs_b)
        if abs(macro_f1(y, swapped_a) - macro_f1(y, swapped_b)) >= abs(delta) - 1e-12:
            count += 1
    return delta, (count + 1) / (n + 1)


out = {"configs": {}, "contrasts": []}
for path in sorted(glob.glob(os.path.join(HERE, "preds_*_s0.npz"))):
    key = os.path.basename(path)[6:-7]
    data = np.load(path)
    lo, hi = bootstrap_ci(data['y'], data['probs'])
    out["configs"][key] = {"test_macro_f1": round(macro_f1(data['y'], data['probs']), 4), "ci95": [round(lo, 4), round(hi, 4)]}
    print(key, out["configs"][key])
# narrative against synopsis, paired by record, per embedding family, all seeds
contrasts = []
for emb in ("tfidf", "glove200", "w2vview"):
    seeds = (0,) if emb == "tfidf" else (0, 1, 2)
    for seed in seeds:
        narr_path = os.path.join(HERE, f"preds_narr_{emb}_s{seed}.npz")
        syn_path = os.path.join(HERE, f"preds_syn_{emb}_s{seed}.npz")
        if not (os.path.exists(narr_path) and os.path.exists(syn_path)):
            continue
        narr_data, syn_data = np.load(narr_path), np.load(syn_path)
        assert list(narr_data['acns']) == list(syn_data['acns'])
        delta, p = paired(narr_data['y'], narr_data['probs'], syn_data['probs'])
        contrasts.append({"family": "view", "contrast": f"{emb} s{seed}: narr vs syn", "delta_f1": round(delta, 4), "p": round(p, 4)})
# reporter 1 against reporter 2, narrative models
for path in sorted(glob.glob(os.path.join(HERE, "dualpreds_*_s*.npz"))):
    data = np.load(path)
    delta, p = paired(data['y'], data['p1'], data['p2'])
    contrasts.append({"family": "author", "contrast": os.path.basename(path)[10:-4] + ": r1 vs r2", "delta_f1": round(delta, 4),
                      "p": round(p, 4), "n": int(len(data['y']))})
m = len(contrasts)
for i, contrast in enumerate(sorted(contrasts, key=lambda c: c["p"])):
    contrast["p_holm"] = round(min(1.0, contrast["p"] * (m - i)), 4)
out["contrasts"] = contrasts
json.dump(out, open(os.path.join(HERE, 'records_stats.json'), 'w'), indent=1)
print(json.dumps(contrasts, indent=1))
