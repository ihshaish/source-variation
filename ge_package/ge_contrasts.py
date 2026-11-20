"""Paired statistics for the contrasts listed in ge_specification.md, from the
held-out prediction arrays. For each contrast A minus B the estimate is the
mean over the three trainings of the within-training macro-F1 difference.
The 95% interval is a paired bootstrap of 10,000 draws, each draw shared by
both conditions and all trainings, repeated with units resampled in place of
events. The p-value is a two-sided approximate randomisation test of 10,000
permutations, one swap per event shared across trainings, plus-one
corrected and Holm-adjusted within each family. Configurations tagged
"orig" are read from results/ (ge_queue.py), "new" from results_contrasts/
(ge_queue_contrasts.py). Also writes per-training confusion counts and
per-assembly miss and false-alert counts, and checks that every prediction
file is in held-out order. Writes contrasts.json and contrasts_table.csv to
results_contrasts/. Run: python ge_contrasts.py [--quick] (500 draws)
"""
import csv
import hashlib
import json
import os
import sys

import numpy as np

from ge_lexicon import OUT, load_lexicon
from ge_lib import DATA, RES, load_records, macro_f1

QUICK = "--quick" in sys.argv
B = 500 if QUICK else 10_000
P = 500 if QUICK else 10_000
K = 4
FULL = "_mask-outcome_terms_v2"
NAMES = "_mask-assembly_names_v2"
FOLDERS = {"orig": RES, "new": OUT}

# id, family, description, A, B, folder of A, folder of B
CONTRASTS = [
    ("H1", "headline", "technician minus customer, Avi2Vec-BiLSTM",
     "random_technician_avi2vec_bilstm", "random_customer_avi2vec_bilstm", "orig", "orig"),
    ("H2", "headline", "repair action: GloVe BiLSTM minus GloVe mean pooling",
     "random_repair_glove200_bilstm", "random_repair_glove200_meanmlp", "orig", "orig"),
    ("M1", "model_by_field", "customer: GloVe BiLSTM minus GloVe mean pooling",
     "random_customer_glove200_bilstm", "random_customer_glove200_meanmlp", "new", "new"),
    ("M2", "model_by_field", "technician: GloVe BiLSTM minus GloVe mean pooling",
     "random_technician_glove200_bilstm", "random_technician_glove200_meanmlp", "new", "new"),
    ("M3", "model_by_field", "repair action: GloVe BiLSTM minus GloVe mean pooling, retrained",
     "random_repair_glove200_bilstm", "random_repair_glove200_meanmlp", "new", "new"),
    ("K1", "masking", "technician: plain minus full mask",
     "random_technician_avi2vec_bilstm", "random_technician_avi2vec_bilstm" + FULL, "new", "new"),
    ("K2", "masking", "technician: plain minus names-only mask",
     "random_technician_avi2vec_bilstm", "random_technician_avi2vec_bilstm" + NAMES, "new", "new"),
    ("K3", "masking", "technician with full mask minus plain customer",
     "random_technician_avi2vec_bilstm" + FULL, "random_customer_avi2vec_bilstm", "new", "new"),
    ("K4", "masking", "technician minus customer, full mask on both",
     "random_technician_avi2vec_bilstm" + FULL, "random_customer_avi2vec_bilstm" + FULL, "new", "new"),
    ("K5", "masking", "technician with names-only mask minus plain customer",
     "random_technician_avi2vec_bilstm" + NAMES, "random_customer_avi2vec_bilstm", "new", "new"),
    ("R1", "lineage", "technician minus customer, Avi2Vec-BiLSTM, retrained",
     "random_technician_avi2vec_bilstm", "random_customer_avi2vec_bilstm", "new", "new"),
    ("S1", "split", "technician minus customer on the chronological split",
     "temporal_technician_avi2vec_bilstm", "temporal_customer_avi2vec_bilstm", "new", "new"),
    ("S2", "split", "technician minus customer on the unit-grouped split",
     "unit_grouped_technician_avi2vec_bilstm", "unit_grouped_customer_avi2vec_bilstm", "new", "new"),
    ("S3", "split", "repair action: plain minus full mask",
     "random_repair_avi2vec_bilstm", "random_repair_avi2vec_bilstm" + FULL, "new", "new"),
]
# difference of two masking drops, (T - T masked) - (C - C masked); interval only
JOINT = ("J1", "masking", "technician masking drop minus customer masking drop, full mask",
         ["random_technician_avi2vec_bilstm", "random_technician_avi2vec_bilstm" + FULL,
          "random_customer_avi2vec_bilstm", "random_customer_avi2vec_bilstm" + FULL])

records = load_records()
splits = json.load(open(os.path.join(DATA, "splits.json")))
with open(os.path.join(DATA, "ge_records.jsonl.gz"), "rb") as f:
    records_sha = hashlib.sha256(f.read()).hexdigest()
class_parts = load_lexicon()[3]


def test_set(split):
    """Held-out records in the order the trainers write predictions."""
    ids = set(splits[split])
    rows = [r for r in records if r["id"] in ids]
    order_sha = hashlib.sha256("\n".join(r["id"] for r in rows).encode()).hexdigest()
    return rows, np.array([int(r["label"]) for r in rows]), order_sha


def load_runs(key, source):
    split = key.split("_technician")[0].split("_customer")[0].split("_repair")[0]
    rows, y_true, order_sha = test_set(split)
    runs = []
    checks = []
    for s in (0, 1, 2):
        path = os.path.join(FOLDERS[source], f"preds_{key}_s{s}.npz")
        if not os.path.exists(path):
            return None, None, None, [f"{source}:{key}_s{s} missing"]
        d = np.load(path)
        if len(d["y"]) != len(y_true) or not np.array_equal(d["y"], y_true):
            status = "FAILED: labels differ from the held-out set in file order"
        elif "order_sha" in d.files:
            if str(d["order_sha"]) == order_sha:
                status = "passed (ID order hash)"
            else:
                status = "FAILED: ID order hash"
        else:
            status = "labels match in held-out order"
        checks.append(f"{source}:{key}_s{s} {status}")
        runs.append(d["yhat"].astype(int))
    return rows, y_true, runs, checks


def counts_matrix(y, yhat):
    """One-hot of the 16 confusion cells per event, shape (n, 16)."""
    return np.eye(K * K, dtype=np.float64)[y * K + yhat]


def f1_from_counts(c):
    """Macro-F1 over the four classes from confusion counts (..., 16)."""
    c = c.reshape(c.shape[:-1] + (K, K))
    tp = np.diagonal(c, axis1=-2, axis2=-1)
    fp = c.sum(-2) - tp
    fn = c.sum(-1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1), 0.0)
    return f1.mean(-1)


def weights(rows, rng, by_unit):
    """B rows of resampling counts per event, drawn over events or over units."""
    n = len(rows)
    if not by_unit:
        return rng.multinomial(n, np.full(n, 1 / n), size=B).astype(np.float64), n
    units = [r.get("unit") or r["id"] for r in rows]
    uniq, idx = np.unique(units, return_inverse=True)
    draws = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=B).astype(np.float64)
    return draws[:, idx], len(uniq)


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def describe_config(key, source, y, runs):
    """Per-training macro-F1, confusion counts and per-assembly counts."""
    name = f"{source}:{key}"
    if name in out["configs"]:
        return
    entry = {"macro_f1_runs": [], "confusion_runs": [], "assembly_rates_runs": []}
    for yhat in runs:
        entry["macro_f1_runs"].append(round(float(macro_f1(y, yhat)), 4))
        entry["confusion_runs"].append(np.bincount(y * K + yhat, minlength=K * K).reshape(K, K).tolist())
        rates = {}
        for part in sorted(set().union(*class_parts.values())):
            has = np.array([part in class_parts[c] for c in range(K)])
            truth = has[y]
            pred = has[yhat]
            rates[part] = {"missed": int((truth & ~pred).sum()), "replaced": int(truth.sum()),
                           "false_alert": int((~truth & pred).sum()), "not_replaced": int((~truth).sum())}
        entry["assembly_rates_runs"].append(rates)
    entry["macro_f1_mean"] = round(float(np.mean(entry["macro_f1_runs"])), 4)
    out["configs"][name] = entry


rng = np.random.default_rng(20261115)
out = {"settings": {"bootstrap_draws": B, "permutations": P, "quick": QUICK,
                    "records_file_sha256": records_sha[:16]},
       "alignment": [], "contrasts": [], "configs": {}, "skipped": []}

for cid, family, desc, key_a, key_b, source_a, source_b in CONTRASTS:
    rows, y, runs_a, checks_a = load_runs(key_a, source_a)
    _, y_b, runs_b, checks_b = load_runs(key_b, source_b)
    out["alignment"] += checks_a + checks_b
    if runs_a is None or runs_b is None:
        out["skipped"].append(cid)
        continue
    if not np.array_equal(y, y_b):
        out["skipped"].append(cid + " (different held-out sets)")
        continue
    describe_config(key_a, source_a, y, runs_a)
    describe_config(key_b, source_b, y, runs_b)
    counts_a = [counts_matrix(y, r) for r in runs_a]
    counts_b = [counts_matrix(y, r) for r in runs_b]
    run_diffs = [f1_from_counts(a.sum(0)) - f1_from_counts(b.sum(0)) for a, b in zip(counts_a, counts_b)]
    point = np.mean(run_diffs)
    res = {"id": cid, "family": family, "contrast": desc,
           "A": f"{source_a}:{key_a}", "B": f"{source_b}:{key_b}",
           "n_events": len(y), "estimate": round(float(point), 4),
           "run_differences": [round(float(d), 4) for d in run_diffs]}
    for label, by_unit in (("ci95", False), ("ci95_unit_cluster", True)):
        W, n_groups = weights(rows, rng, by_unit)
        diffs = np.mean([f1_from_counts(W @ a) - f1_from_counts(W @ b)
                         for a, b in zip(counts_a, counts_b)], axis=0)
        res[label] = [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)]
        if by_unit:
            res["n_units"] = int(n_groups)
    S = (rng.random((P, len(y))) < 0.5).astype(np.float64)
    perm = np.mean([f1_from_counts((1 - S) @ a + S @ b) - f1_from_counts(S @ a + (1 - S) @ b)
                    for a, b in zip(counts_a, counts_b)], axis=0)
    res["p_raw"] = float((1 + np.sum(np.abs(perm) >= abs(point) - 1e-12)) / (1 + P))
    out["contrasts"].append(res)

cid, family, desc, keys = JOINT
loaded = [load_runs(k, "new") for k in keys]
if all(item[2] is not None for item in loaded) and all(np.array_equal(loaded[0][1], item[1]) for item in loaded):
    rows, y = loaded[0][0], loaded[0][1]
    counts = [[counts_matrix(y, r) for r in item[2]] for item in loaded]
    sign = (1, -1, -1, 1)
    point = np.mean([sum(sg * f1_from_counts(counts[c][s].sum(0)) for c, sg in enumerate(sign))
                     for s in range(3)])
    W, _ = weights(rows, rng, False)
    diffs = np.mean([sum(sg * f1_from_counts(W @ counts[c][s]) for c, sg in enumerate(sign))
                     for s in range(3)], axis=0)
    out["contrasts"].append({"id": cid, "family": family, "contrast": desc, "n_events": len(y),
                             "estimate": round(float(point), 4),
                             "ci95": [round(float(np.percentile(diffs, 2.5)), 4),
                                      round(float(np.percentile(diffs, 97.5)), 4)]})
else:
    out["skipped"].append(cid)

for fam in {c["family"] for c in out["contrasts"]}:
    members = [c for c in out["contrasts"] if c["family"] == fam and "p_raw" in c]
    if members:
        for c, adjusted in zip(members, holm(np.array([m["p_raw"] for m in members]))):
            c["p_holm_within_family"] = round(float(adjusted), 4)
            c["p_raw"] = round(c["p_raw"], 4)

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "contrasts.json"), "w") as f:
    json.dump(out, f, indent=1)
with open(os.path.join(OUT, "contrasts_table.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "family", "contrast", "n_events", "estimate", "ci_low", "ci_high",
                     "unit_ci_low", "unit_ci_high", "n_units", "p_raw", "p_holm", "run_differences"])
    for c in out["contrasts"]:
        writer.writerow([c["id"], c["family"], c["contrast"], c["n_events"], c["estimate"], *c["ci95"],
                         *c.get("ci95_unit_cluster", ["", ""]), c.get("n_units", ""), c.get("p_raw", ""),
                         c.get("p_holm_within_family", ""), " ".join(map(str, c.get("run_differences", [])))])

print(f"{'id':4s} {'estimate':>9s}  {'95% CI':>18s}  {'unit CI':>18s}  {'p_holm':>7s}  contrast")
for c in out["contrasts"]:
    unit_ci = c.get("ci95_unit_cluster")
    unit_text = "[%+.4f, %+.4f]" % tuple(unit_ci) if unit_ci else " " * 18
    print(f"{c['id']:4s} {c['estimate']:+9.4f}  [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  "
          f"{unit_text}  {c.get('p_holm_within_family', ''):>7}  {c['contrast']}")
print("\nskipped:", ", ".join(out["skipped"]) or "none")
failed = [a for a in out["alignment"] if "FAILED" in a]
print("alignment:", "all passed" if not failed else "\n  " + "\n  ".join(failed))
