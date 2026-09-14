"""The comparable-length rows of the dual-report analysis, per embedding and
training: cases whose supplemental narrative has at least 0.7 of the primary
narrative's whitespace-split length, primary-minus-supplemental macro-F1 and
a paired approximate-randomisation p-value. Reads records_task.jsonl.gz and
dualpreds_*_s*.npz; writes dual_matched_runs.json.

    python3 dual_matched.py
"""
import gzip
import json
import os

import numpy as np
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(20260802)


def macro_f1(y, predicted):
    return f1_score(y, predicted, average="macro")


def paired_test(y, predicted_a, predicted_b, n_permutations=10000):
    observed = macro_f1(y, predicted_a) - macro_f1(y, predicted_b)
    count = 0
    for _ in range(n_permutations):
        swap = rng.random(len(y)) < 0.5
        permuted_a = np.where(swap, predicted_b, predicted_a)
        permuted_b = np.where(swap, predicted_a, predicted_b)
        if abs(macro_f1(y, permuted_a) - macro_f1(y, permuted_b)) >= abs(observed) - 1e-12:
            count += 1
    return observed, (count + 1) / (n_permutations + 1)


records = [json.loads(line) for line in gzip.open(os.path.join(HERE, "records_task.jsonl.gz"), "rt")]
by_acn = {record["acn"]: record for record in records if record.get("r2")}

first = np.load(os.path.join(HERE, "dualpreds_w2vview_s0.npz"))
acns = first["acns"]
y = first["y"]
ratio = np.array([len(by_acn[acn]["r2"].split()) / max(1, len(by_acn[acn]["narr"].split())) for acn in acns])
mask = ratio >= 0.7
y_matched = y[mask]

rows = []
for embedding in ("glove200", "w2vview"):
    for training in (0, 1, 2):
        d = np.load(os.path.join(HERE, f"dualpreds_{embedding}_s{training}.npz"))
        predicted_primary = (d["p1"] >= 0.5).astype(int)
        predicted_supplemental = (d["p2"] >= 0.5).astype(int)
        delta, p = paired_test(y_matched, predicted_primary[mask], predicted_supplemental[mask])
        rows.append({"emb": embedding, "run": training + 1, "delta": round(float(delta), 4), "p": round(p, 4)})
        print(rows[-1])
json.dump(rows, open(os.path.join(HERE, "dual_matched_runs.json"), "w"), indent=1)
