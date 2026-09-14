"""Coverage bound on the held-out records. Groups the test records of the
random split whose encoded token sequences are identical under a given
embedding vocabulary, and bounds accuracy by the majority label of each
group. The bound is most informative on the short, templated repair-action
field, where duplicate narratives make collisions certain. Reads
ge_records.jsonl.gz and splits.json from GE_DATA and writes
covbound_report.json to GE_RES. Run: python ge_covbound.py
"""
import json
import os

import numpy as np

from ge_lib import DATA, build_matrix, encode, load_records, record_tokens

records = load_records()
split = set(json.load(open(os.path.join(DATA, "splits.json")))["random"])
test = [record for record in records if record["id"] in split]
out = {}
for field in ("customer", "technician", "repair"):
    vocab = set()
    for record in records:
        vocab.update(record_tokens(record, field))
    for emb in ("glove200", "avi2vec"):
        index, _ = build_matrix(emb, vocab)
        X, y = encode(test, field, index)
        groups = {}
        for i, row in enumerate(X):
            groups.setdefault(row.tobytes(), []).append(i)
        colliding = [members for members in groups.values() if len(members) > 1]
        conflicting = [members for members in colliding
                       if len({int(y[j]) for j in members}) > 1]
        bound = sum(int(max(np.bincount([y[j] for j in members])))
                    for members in groups.values()) / len(y)
        out[f"{field}_{emb}"] = {
            "test_records": len(y),
            "colliding_records": int(sum(len(members) for members in colliding)),
            "label_conflicting_records": int(sum(len(members) for members in conflicting)),
            "accuracy_bound": round(bound, 4)}
        print(f"{field}/{emb}: bound {bound:.4f}, "
              f"colliding {sum(len(members) for members in colliding)}, "
              f"conflicting {sum(len(members) for members in conflicting)}")
json.dump(out, open(os.path.join(os.environ.get('GE_RES', 'results'), "covbound_report.json"), "w"),
          indent=1)
