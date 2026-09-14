"""Differential-vocabulary stratification and masking test. With --make-mask
it writes masks/vdelta.txt, the task tokens that are in the Avi2Vec
vocabulary but not in GloVe-200; ge_train.py is then run with and without
that mask file. With --analyse it splits the held-out records of the random
split into those containing at least one masked token and those containing
none, reports macro-F1 per stratum for every prediction file of the field,
and writes strata_report.json to GE_RES.
Run: python ge_strata.py --make-mask | --analyse [--field repair]
"""
import argparse
import glob
import json
import os

import numpy as np

from ge_lib import DATA, RES, load_records, macro_f1, record_tokens, build_matrix

parser = argparse.ArgumentParser()
parser.add_argument("--make-mask", action="store_true")
parser.add_argument("--analyse", action="store_true")
parser.add_argument("--field", default="repair")
args = parser.parse_args()
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "masks"), exist_ok=True)
mask_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "masks", "vdelta.txt")

records = load_records()
task_vocab = set()
for record in records:
    task_vocab.update(record_tokens(record, args.field))

if args.make_mask:
    avi_index, _ = build_matrix("avi2vec", task_vocab)
    glove_index, _ = build_matrix("glove200", task_vocab)
    vdelta = sorted(set(avi_index) - set(glove_index))
    reverse = sorted(set(glove_index) - set(avi_index))
    open(mask_path, "w").write("\n".join(vdelta))
    print(f"V_delta: {len(vdelta)} tokens, written to {mask_path}")
    print(f"reverse differential (GloVe minus Avi2Vec, task tokens): {len(reverse)}")

if args.analyse:
    vdelta = {line.strip() for line in open(mask_path) if line.strip()}
    split = set(json.load(open(os.path.join(DATA, "splits.json")))["random"])
    test = [record for record in records if record["id"] in split]
    has_vdelta = np.array([any(token in vdelta for token in record_tokens(record, args.field))
                           for record in test])
    print(f"stratum sizes: V_delta-positive {has_vdelta.sum()}, negative {(~has_vdelta).sum()}")
    report = {}
    for path in sorted(glob.glob(os.path.join(RES, f"preds_random_{args.field}_*_s0.npz"))):
        data = np.load(path)
        y = data["y"]
        yhat = data["yhat"]
        if len(y) != len(test):
            print(f"skip {os.path.basename(path)} (size mismatch)")
            continue
        report[os.path.basename(path)] = {
            "overall": round(macro_f1(y, yhat), 4),
            "vdelta_pos": round(macro_f1(y[has_vdelta], yhat[has_vdelta]), 4),
            "vdelta_neg": round(macro_f1(y[~has_vdelta], yhat[~has_vdelta]), 4)}
    json.dump(report, open(os.path.join(RES, "strata_report.json"), "w"), indent=1)
    print(json.dumps(report, indent=1))
