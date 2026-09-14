"""Statistics on the held-out predictions of the random split: macro-F1 with
bootstrap 95% confidence intervals, paired approximate-randomisation tests
of every other BiLSTM embedding against the Avi2Vec BiLSTM within each
field, Holm-corrected within the field, plus per-class precision, recall,
F1 and confusion matrices. Runs on whatever prediction files exist. Reads
preds_random_*_s0.npz from GE_RES and writes ge_stats.json there.
Run: python ge_stats.py
"""
import glob
import json
import os

import numpy as np

from ge_lib import RES, macro_f1

N_RESAMPLES = 10_000
rng = np.random.default_rng(20260802)


def boot_ci(y, yhat):
    n = len(y)
    scores = np.empty(N_RESAMPLES)
    for b in range(N_RESAMPLES):
        sample = rng.integers(0, n, n)
        scores[b] = macro_f1(y[sample], yhat[sample])
    return round(float(np.percentile(scores, 2.5)), 4), round(float(np.percentile(scores, 97.5)), 4)


def rand_p(yhat_a, yhat_b, y):
    observed = abs(macro_f1(y, yhat_a) - macro_f1(y, yhat_b))
    differs = yhat_a != yhat_b
    n = len(y)
    count = 0
    for _ in range(N_RESAMPLES):
        swap = (rng.random(n) < 0.5) & differs
        swapped_a = np.where(swap, yhat_b, yhat_a)
        swapped_b = np.where(swap, yhat_a, yhat_b)
        if abs(macro_f1(y, swapped_a) - macro_f1(y, swapped_b)) >= observed - 1e-12:
            count += 1
    return (count + 1) / (N_RESAMPLES + 1)


def holm(p_values):
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    m = len(p_values)
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def perclass(y, yhat, n_classes=4):
    rows = []
    for label in range(n_classes):
        tp = int(((yhat == label) & (y == label)).sum())
        fp = int(((yhat == label) & (y != label)).sum())
        fn = int(((yhat != label) & (y == label)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        if precision + recall:
            f1 = round(2 * precision * recall / (precision + recall), 3)
        else:
            f1 = 0.0
        rows.append({"class": label, "precision": round(precision, 3), "recall": round(recall, 3),
                     "f1": f1, "support": int((y == label).sum())})
    confusion = [[int(((y == i) & (yhat == j)).sum()) for j in range(n_classes)]
                 for i in range(n_classes)]
    return rows, confusion


preds = {}
for path in glob.glob(os.path.join(RES, "preds_random_*_s0.npz")):
    key = os.path.basename(path)[len("preds_random_"):-len("_s0.npz")]
    data = np.load(path)
    preds[key] = (data["y"], data["yhat"])

report = {"configs": {}, "contrasts": []}
for key, (y, yhat) in sorted(preds.items()):
    rows, confusion = perclass(y, yhat)
    report["configs"][key] = {"test_macro_f1": round(float(macro_f1(y, yhat)), 4),
                              "ci95": list(boot_ci(y, yhat)),
                              "per_class": rows, "confusion": confusion}
    print(key, report["configs"][key]["test_macro_f1"], report["configs"][key]["ci95"])

fields = sorted({key.split("_")[0] for key in preds})
for field in fields:
    family = []
    for key in preds:
        if (key.startswith(field + "_") and "_bilstm" in key
                and "avi2vec" not in key and "mask-" not in key):
            ref_key = f"{field}_avi2vec_bilstm"
            if ref_key in preds:
                family.append((ref_key, key))
    p_values = []
    rows = []
    for ref_key, other_key in family:
        y, yhat_ref = preds[ref_key]
        _, yhat_other = preds[other_key]
        delta = macro_f1(y, yhat_ref) - macro_f1(y, yhat_other)
        p_value = rand_p(yhat_ref, yhat_other, y)
        p_values.append(p_value)
        rows.append({"contrast": f"{ref_key} vs {other_key}",
                     "delta": round(float(delta), 4), "p": p_value})
    for row, p_adjusted in zip(rows, holm(p_values)):
        row["p_holm"] = round(p_adjusted, 4)
        row["p"] = round(row["p"], 4)
        report["contrasts"].append(row)
        print(row)

json.dump(report, open(os.path.join(RES, "ge_stats.json"), "w"), indent=1)
