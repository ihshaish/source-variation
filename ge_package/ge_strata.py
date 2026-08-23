"""P1 — differential-vocabulary stratification + masking test (the registered
mechanism experiment, FIGURES_TODO item 2 / manuscript Sec 5.4).

Step 1 (this script, --make-mask): V_delta = vocab(Avi2Vec) minus
vocab(GloVe-200), intersected with the task vocabulary -> masks/vdelta.txt.
Step 2 (ge_train, run 4x): repair-field BiLSTM under avi2vec and glove200,
each with and without --mask-file masks/vdelta.txt.
Step 3 (this script, --analyse): per-stratum macro-F1 (records containing >=1
V_delta token vs none) for every available prediction file, plus the masked
vs unmasked contrast. Registered predictions: the embedding gap concentrates
in the V_delta-positive stratum and collapses under masking.
"""
import argparse, glob, json, os, re
import numpy as np
from ge_lib import DATA, RES, load_records, macro_f1, record_tokens, build_matrix

ap = argparse.ArgumentParser()
ap.add_argument("--make-mask", action="store_true")
ap.add_argument("--analyse", action="store_true")
ap.add_argument("--field", default="repair")
a = ap.parse_args()
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "masks"), exist_ok=True)
mask_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "masks", "vdelta.txt")

recs = load_records()
task_vocab = set()
for r in recs:
    task_vocab.update(record_tokens(r, a.field))

if a.make_mask:
    idx_a, _ = build_matrix("avi2vec", task_vocab)
    idx_g, _ = build_matrix("glove200", task_vocab)
    vdelta = sorted(set(idx_a) - set(idx_g))
    reverse = sorted(set(idx_g) - set(idx_a))
    open(mask_path, "w").write("\n".join(vdelta))
    print(f"V_delta: {len(vdelta)} tokens -> {mask_path}")
    print(f"reverse differential (GloVe minus Avi2Vec, task tokens): {len(reverse)}")

if a.analyse:
    vdelta = {l.strip() for l in open(mask_path) if l.strip()}
    split = set(json.load(open(os.path.join(DATA, "splits.json")))["random"])
    test = [r for r in recs if r["id"] in split]
    has = np.array([any(t in vdelta for t in record_tokens(r, a.field)) for r in test])
    print(f"stratum sizes: V_delta-positive {has.sum()}, negative {(~has).sum()}")
    report = {}
    for p in sorted(glob.glob(os.path.join(RES, f"preds_random_{a.field}_*_s0.npz"))):
        d = np.load(p); y, yhat = d["y"], d["yhat"]
        if len(y) != len(test):
            print(f"skip {os.path.basename(p)} (size mismatch)"); continue
        report[os.path.basename(p)] = {
            "overall": round(macro_f1(y, yhat), 4),
            "vdelta_pos": round(macro_f1(y[has], yhat[has]), 4),
            "vdelta_neg": round(macro_f1(y[~has], yhat[~has]), 4)}
    json.dump(report, open(os.path.join(RES, "strata_report.json"), "w"), indent=1)
    print(json.dumps(report, indent=1))
