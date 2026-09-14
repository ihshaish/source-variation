"""Length analysis of the dual-report contrast, from the stored dual-report
predictions. Bins the held-out dual-report cases by the length of the
supplemental narrative and gives the primary-minus-supplemental difference
per bin, then runs paired tests on the subset whose supplemental narrative
has at least 0.7 of the primary narrative's tokens. Also counts records,
median tokens and vocabulary for the three records. Reads records_task.jsonl.gz
and dualpreds_*_s*.npz; writes dual_report_length.json and
records_characterisation.json.

    python3 dual_report_length.py
"""
import glob
import gzip
import json
import os
import re
import statistics

import numpy as np
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
TOKEN_RE = re.compile(r"[a-z][a-z0-9/-]+")
rng = np.random.default_rng(20260802)

records = [json.loads(line) for line in gzip.open(os.path.join(HERE, "records_task.jsonl.gz"), "rt")]


def macro_f1(y, probs):
    return f1_score(y, (probs >= 0.5).astype(int), average="macro")


def paired_test(y, probs_a, probs_b, n_permutations=10000):
    observed = macro_f1(y, probs_a) - macro_f1(y, probs_b)
    count = 0
    for _ in range(n_permutations):
        swap = rng.random(len(y)) < 0.5
        permuted = macro_f1(y, np.where(swap, probs_b, probs_a)) - macro_f1(y, np.where(swap, probs_a, probs_b))
        if abs(permuted) >= abs(observed) - 1e-12:
            count += 1
    return observed, (count + 1) / (n_permutations + 1)


lengths = {}
for record in records:
    if record["split"] == "test" and len(record["r2"]) > 40:
        primary = len(TOKEN_RE.findall(record["narr"].lower()))
        supplemental = len(TOKEN_RE.findall(record["r2"].lower()))
        lengths[record["acn"]] = (primary, supplemental)

result = {"dose": [], "matched": []}

bins = [(0, 25), (25, 60), (60, 100), (100, 150), (150, 10**6)]
for low, high in bins:
    deltas = []
    for path in sorted(glob.glob(os.path.join(HERE, "dualpreds_w2vview_s*.npz"))):
        d = np.load(path)
        supplemental = np.array([lengths[acn][1] for acn in d["acns"]])
        mask = (supplemental >= low) & (supplemental < high)
        if mask.sum() >= 60:
            deltas.append(macro_f1(d["y"][mask], d["p1"][mask]) - macro_f1(d["y"][mask], d["p2"][mask]))
    if deltas:
        label = f"{low}-{high if high < 10**6 else 'max'}"
        result["dose"].append({"r2_tokens": label, "n": int(mask.sum()),
                               "delta_mean": round(float(np.mean(deltas)), 3),
                               "deltas": [round(x, 3) for x in deltas]})
        print(result["dose"][-1])

for path in sorted(glob.glob(os.path.join(HERE, "dualpreds_*_s*.npz"))):
    d = np.load(path)
    primary = np.array([lengths[acn][0] for acn in d["acns"]])
    supplemental = np.array([lengths[acn][1] for acn in d["acns"]])
    mask = supplemental >= 0.7 * primary
    delta, p = paired_test(d["y"][mask], d["p1"][mask], d["p2"][mask])
    model = os.path.basename(path)[len("dualpreds_"):-len(".npz")]
    result["matched"].append({"model": model, "n": int(mask.sum()), "delta": round(delta, 4), "p": round(p, 4)})
    print(result["matched"][-1])

json.dump(result, open(os.path.join(HERE, "dual_report_length.json"), "w"), indent=1)

characterisation = {}
for key in ("narr", "syn", "r2"):
    token_lists = []
    for record in records:
        if record[key].strip() and (key != "r2" or len(record[key]) > 40):
            token_lists.append(TOKEN_RE.findall(record[key].lower()))
    vocabulary = set(token for tokens in token_lists for token in tokens)
    characterisation[key] = {"records": len(token_lists),
                             "median_tokens": int(statistics.median(len(tokens) for tokens in token_lists)),
                             "vocab": len(vocabulary)}
    print(key, characterisation[key])
json.dump(characterisation, open(os.path.join(HERE, "records_characterisation.json"), "w"), indent=1)
