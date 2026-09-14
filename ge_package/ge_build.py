"""Builds the record file and every split scheme from the raw records CSV.

Reads ge_data/ge_records.csv with the columns listed in README.md:
  record_id, date (YYYYMM or YYYY-MM-DD), customer, technician, repair,
  label (0-3 or a class name mapped through label_map.json), unit_serial
  (optional), operator (optional)

Writes ge_data/ge_records.jsonl.gz and ge_data/splits.json, which holds the
held-out test record_ids for each scheme:
  random           stratified 80/20 (the default split)
  dup_exact        identical repair-action texts kept together
  dup_near         near-duplicates (minhash, verified Jaccard >= 0.8) kept together
  unit_grouped     all records of one unit_serial on one side (if the column is present)
  operator_grouped all records of one operator on one side (if the column is present)
  temporal         train = earliest 80% by date, test = latest 20%

Run: python ge_build.py
"""
import csv
import gzip
import json
import os
from collections import Counter

import numpy as np

from ge_lib import DATA, GLOBAL_SEED, tokenize

LABEL_MAP_PATH = os.path.join(DATA, "label_map.json")


def exact_groups(texts):
    """Groups records whose normalised token sequences are identical."""
    seen = {}
    out = []
    for text in texts:
        key = " ".join(tokenize(text))
        out.append(seen.setdefault(key, len(seen)))
    return out


def near_dup_groups(texts, shingle=5, jaccard=0.8, perms=64, bands=16):
    """Groups near-duplicate texts. Candidate pairs come from minhash LSH and
    are joined by union-find only when their verified Jaccard overlap is at
    least `jaccard`; requiring a high verified overlap stops templated text
    collapsing into one giant group."""
    rng = np.random.default_rng(0)
    prime = (1 << 61) - 1
    coeff = rng.integers(1, prime, size=(perms, 2), dtype=np.int64)
    shingle_sets = []
    signatures = np.full((len(texts), perms), np.iinfo(np.int64).max)
    for i, text in enumerate(texts):
        tokens = tokenize(text)
        shingles = {hash(" ".join(tokens[j:j + shingle])) & 0x7FFFFFFFFFFFFFFF
                    for j in range(max(1, len(tokens) - shingle + 1))} or {hash(text)}
        shingle_sets.append(shingles)
        values = np.fromiter(shingles, dtype=np.int64)
        for perm in range(perms):
            signatures[i, perm] = int(((coeff[perm, 0] * values + coeff[perm, 1]) % prime).min())
    parent = list(range(len(texts)))

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    rows = perms // bands
    for band in range(bands):
        buckets = {}
        for i in range(len(texts)):
            key = signatures[i, band * rows:(band + 1) * rows].tobytes()
            buckets.setdefault(key, []).append(i)
        for members in buckets.values():
            for j in members[1:]:
                first = shingle_sets[members[0]]
                other = shingle_sets[j]
                if len(first & other) / len(first | other) >= jaccard:
                    parent[find(j)] = find(members[0])
    return [find(i) for i in range(len(texts))]


def stratified_take(labels, frac, rng, groups=None):
    """Returns a set of indices holding about `frac` of each class; when
    groups are given, whole groups move together."""
    labels = np.asarray(labels)
    take = set()
    if groups is None:
        for label in np.unique(labels):
            indices = np.where(labels == label)[0]
            indices = indices[rng.permutation(len(indices))]
            take.update(indices[: round(frac * len(indices))].tolist())
        return take
    groups = np.asarray(groups)
    shuffled = rng.permutation(np.unique(groups))
    target = frac * len(labels)
    for group in shuffled:
        if len(take) >= target:
            break
        take.update(np.where(groups == group)[0].tolist())
    return take


def main():
    label_map = json.load(open(LABEL_MAP_PATH)) if os.path.exists(LABEL_MAP_PATH) else None
    records = []
    with open(os.path.join(DATA, "ge_records.csv"), newline="", encoding="utf-8",
              errors="replace") as f:
        for row in csv.DictReader(f):
            label = row["label"].strip()
            label = int(label) if label.isdigit() else label_map[label]
            records.append({"id": row["record_id"], "date": row.get("date", ""),
                            "customer": row.get("customer", ""),
                            "technician": row.get("technician", ""),
                            "repair": row.get("repair", ""), "label": label,
                            "unit": row.get("unit_serial", ""),
                            "operator": row.get("operator", "")})
    print(f"{len(records)} records")
    with gzip.open(os.path.join(DATA, "ge_records.jsonl.gz"), "wt") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    rng = np.random.default_rng(GLOBAL_SEED)
    labels = [record["label"] for record in records]
    splits = {}
    splits["random"] = stratified_take(labels, 0.2, rng)
    for name, grouper in [("dup_exact", exact_groups), ("dup_near", near_dup_groups)]:
        groups = grouper([record["repair"] for record in records])
        biggest = Counter(groups).most_common(1)[0][1]
        if biggest > 0.5 * len(records):
            print(f"split {name}: degenerate (largest group {biggest}/{len(records)}), skipped")
            continue
        splits[name] = stratified_take(labels, 0.2, rng, groups=groups)
    if any(record["unit"] for record in records):
        splits["unit_grouped"] = stratified_take(
            labels, 0.2, rng, groups=[record["unit"] or record["id"] for record in records])
    if any(record["operator"] for record in records):
        splits["operator_grouped"] = stratified_take(
            labels, 0.2, rng, groups=[record["operator"] or record["id"] for record in records])
    order = np.argsort([record["date"] for record in records], kind="stable")
    splits["temporal"] = set(order[-round(0.2 * len(records)):].tolist())

    out = {name: sorted(records[i]["id"] for i in indices) for name, indices in splits.items()}
    with open(os.path.join(DATA, "splits.json"), "w") as f:
        json.dump(out, f)
    for name, ids in out.items():
        print(f"split {name}: {len(ids)} test records")


if __name__ == "__main__":
    main()
