"""Empirical coverage bound (manuscript Eq. covbound): partition held-out
records into equivalence classes of identical encoded sequences under a given
embedding vocabulary, and bound accuracy by each class's majority label.
Informative on the short, templated repair-action field, where duplicate
narratives guarantee collisions.  Usage: python ge_covbound.py
"""
import json
import os

import numpy as np

from ge_lib import DATA, build_matrix, encode, load_records, record_tokens

recs = load_records()
split = set(json.load(open(os.path.join(DATA, "splits.json")))["random"])
test = [r for r in recs if r["id"] in split]
out = {}
for field in ("customer", "technician", "repair"):
    vocab = set()
    for r in recs:
        vocab.update(record_tokens(r, field))
    for emb in ("glove200", "avi2vec"):
        idx, _ = build_matrix(emb, vocab)
        X, y = encode(test, field, idx)
        groups = {}
        for i, row in enumerate(X):
            groups.setdefault(row.tobytes(), []).append(i)
        multi = [v for v in groups.values() if len(v) > 1]
        conflicts = [v for v in multi if len({int(y[j]) for j in v}) > 1]
        bound = sum(int(max(np.bincount([y[j] for j in v]))) for v in groups.values()) / len(y)
        out[f"{field}_{emb}"] = {
            "test_records": len(y),
            "colliding_records": int(sum(len(v) for v in multi)),
            "label_conflicting_records": int(sum(len(v) for v in conflicts)),
            "accuracy_bound": round(bound, 4)}
        print(f"{field}/{emb}: bound {bound:.4f}, "
              f"colliding {sum(len(v) for v in multi)}, "
              f"conflicting {sum(len(v) for v in conflicts)}")
json.dump(out, open(os.path.join(os.environ.get('GE_RES', 'results'), "covbound_report.json"), "w"), indent=1)
