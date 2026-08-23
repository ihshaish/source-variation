"""Primary statistics stage on GE held-out predictions: bootstrap CIs, paired
approximate-randomisation with Holm within families, per-class P/R/F1 and
confusion matrices (covers plan items P4 + P5). Runs on whatever prediction
files exist; families are inferred from available configs on the random split.
"""
import glob, json, os, re
import numpy as np
from ge_lib import RES, macro_f1

B = 10_000
rng = np.random.default_rng(20260802)

def boot_ci(y, yhat):
    n = len(y); s = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        s[b] = macro_f1(y[i], yhat[i])
    return round(float(np.percentile(s, 2.5)), 4), round(float(np.percentile(s, 97.5)), 4)

def rand_p(a, b, y):
    obs = abs(macro_f1(y, a) - macro_f1(y, b)); diff = a != b
    n = len(y); c = 0
    for _ in range(B):
        s = (rng.random(n) < 0.5) & diff
        if abs(macro_f1(y, np.where(s, b, a)) - macro_f1(y, np.where(s, a, b))) >= obs - 1e-12:
            c += 1
    return (c + 1) / (B + 1)

def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); m = len(ps)
    adj, run = [0.0]*m, 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj

def perclass(y, yhat, k=4):
    rows = []
    for c in range(k):
        tp = int(((yhat == c) & (y == c)).sum()); fp = int(((yhat == c) & (y != c)).sum())
        fn = int(((yhat != c) & (y == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
        rows.append({"class": c, "precision": round(p, 3), "recall": round(r, 3),
                     "f1": round(2*p*r/(p+r), 3) if p + r else 0.0, "support": int((y == c).sum())})
    cm = [[int(((y == i) & (yhat == j)).sum()) for j in range(k)] for i in range(k)]
    return rows, cm

preds = {}
for p in glob.glob(os.path.join(RES, "preds_random_*_s0.npz")):
    key = os.path.basename(p)[len("preds_random_"):-len("_s0.npz")]
    d = np.load(p)
    preds[key] = (d["y"], d["yhat"])

report = {"configs": {}, "contrasts": []}
for key, (y, yhat) in sorted(preds.items()):
    rows, cm = perclass(y, yhat)
    report["configs"][key] = {"test_macro_f1": round(float(macro_f1(y, yhat)), 4),
                              "ci95": list(boot_ci(y, yhat)),
                              "per_class": rows, "confusion": cm}
    print(key, report["configs"][key]["test_macro_f1"], report["configs"][key]["ci95"])

# families: per field, avi2vec vs each other embedding (fixed arch); per field, arch pairs
fields = sorted({k.split("_")[0] for k in preds})
for field in fields:
    fam = []
    for k in preds:
        if k.startswith(field + "_") and "_bilstm" in k and "avi2vec" not in k and "mask-" not in k:
            ref = f"{field}_avi2vec_bilstm"
            if ref in preds:
                fam.append((ref, k))
    ps, rows = [], []
    for r1, r2 in fam:
        y, a = preds[r1]; _, b = preds[r2]
        d = macro_f1(y, a) - macro_f1(y, b); p = rand_p(a, b, y)
        ps.append(p); rows.append({"contrast": f"{r1} vs {r2}", "delta": round(float(d), 4), "p": p})
    for row, padj in zip(rows, holm(ps)):
        row["p_holm"] = round(padj, 4); row["p"] = round(row["p"], 4)
        report["contrasts"].append(row); print(row)

json.dump(report, open(os.path.join(RES, "ge_stats.json"), "w"), indent=1)
