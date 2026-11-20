"""Lists the forty words most specific to each class in each field, ranked by
smoothed log-odds against the other classes, from the training records of the
random split. It is the starting list for completing assembly_lexicon.json.
Writes private/term_candidates.txt, which holds record vocabulary and stays
with the records. Run: python ge_terms.py
"""
import json
import math
import os
from collections import Counter

from ge_lexicon import FIELDS, PRIVATE
from ge_lib import DATA, load_records, record_tokens

records = load_records()
test_ids = set(json.load(open(os.path.join(DATA, "splits.json")))["random"])
train = [r for r in records if r["id"] not in test_ids]
classes = sorted({r["label"] for r in train})
n_class = Counter(r["label"] for r in train)
os.makedirs(PRIVATE, exist_ok=True)
lines = []
for field in FIELDS:
    doc_freq = {c: Counter() for c in classes}
    for r in train:
        doc_freq[r["label"]].update(set(record_tokens(r, field)))
    total = Counter()
    for c in classes:
        total.update(doc_freq[c])
    for c in classes:
        rest_n = len(train) - n_class[c]
        scored = []
        for token, k in doc_freq[c].items():
            if total[token] < 10:
                continue
            rest_k = total[token] - k
            score = (math.log((k + 0.5) / (n_class[c] - k + 0.5))
                     - math.log((rest_k + 0.5) / (rest_n - rest_k + 0.5)))
            scored.append((score, token, k))
        scored.sort(reverse=True)
        lines.append(f"\nfield={field}  class={c}  (n={n_class[c]})")
        lines.append("token                     in-class share   log-odds")
        for score, token, k in scored[:40]:
            lines.append(f"{token:25s} {100 * k / n_class[c]:6.1f}%        {score:6.2f}")
path = os.path.join(PRIVATE, "term_candidates.txt")
with open(path, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"wrote {path}")
