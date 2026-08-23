"""Build the GE task file and every split scheme from the raw records CSV.

Input: ge_data/ge_records.csv with columns (README.md):
  record_id, date (YYYYMM or YYYY-MM-DD), customer, technician, repair,
  label (0-3 or class name per label_map.json), unit_serial (optional),
  operator (optional)

Outputs in ge_data/: ge_records.jsonl.gz, and splits.json holding the held-out
test record_ids for each scheme:
  random          stratified 80/20 (default split, matches the paper)
  dup_exact       identical repair-action texts kept together
  dup_near        near-duplicates (minhash, verified Jaccard >= 0.8) kept together
  unit_grouped    all records of one unit_serial on one side (if column present)
  operator_grouped (if column present)
  temporal        train = earliest 80% by date, test = latest 20%
"""
import csv
import gzip
import json
import os

import numpy as np

from ge_lib import DATA, GLOBAL_SEED, tokenize

LABEL_MAP_PATH = os.path.join(DATA, "label_map.json")


def exact_groups(texts):
    """Group records whose normalised token sequences are identical."""
    seen = {}
    out = []
    for t in texts:
        key = " ".join(tokenize(t))
        out.append(seen.setdefault(key, len(seen)))
    return out


def near_dup_groups(texts, shingle=5, jaccard=0.8, perms=64, bands=16):
    """Near-duplicate grouping: minhash-LSH candidates verified at
    Jaccard >= `jaccard`, then union-find. Requiring high verified overlap
    (rather than any shared shingle) prevents templated text collapsing into
    one giant component."""
    import numpy as np
    rng = np.random.default_rng(0)
    prime = (1 << 61) - 1
    coeff = rng.integers(1, prime, size=(perms, 2), dtype=np.int64)
    shingle_sets, sigs = [], np.full((len(texts), perms), np.iinfo(np.int64).max)
    for i, t in enumerate(texts):
        toks = tokenize(t)
        sh = {hash(" ".join(toks[j:j + shingle])) & 0x7FFFFFFFFFFFFFFF
              for j in range(max(1, len(toks) - shingle + 1))} or {hash(t)}
        shingle_sets.append(sh)
        vals = np.fromiter(sh, dtype=np.int64)
        for p in range(perms):
            sigs[i, p] = int(((coeff[p, 0] * vals + coeff[p, 1]) % prime).min())
    parent = list(range(len(texts)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    rows = perms // bands
    for b in range(bands):
        buckets = {}
        for i in range(len(texts)):
            key = sigs[i, b * rows:(b + 1) * rows].tobytes()
            buckets.setdefault(key, []).append(i)
        for members in buckets.values():
            for j in members[1:]:
                a, c = shingle_sets[members[0]], shingle_sets[j]
                if len(a & c) / len(a | c) >= jaccard:
                    parent[find(j)] = find(members[0])
    return [find(i) for i in range(len(texts))]


def stratified_take(labels, frac, rng, groups=None):
    """Return a set of indices ~frac per class; whole groups move together."""
    labels = np.asarray(labels)
    take = set()
    if groups is None:
        for c in np.unique(labels):
            ix = np.where(labels == c)[0]
            ix = ix[rng.permutation(len(ix))]
            take.update(ix[: round(frac * len(ix))].tolist())
        return take
    groups = np.asarray(groups)
    uniq = rng.permutation(np.unique(groups))
    target = frac * len(labels)
    for g in uniq:
        if len(take) >= target:
            break
        take.update(np.where(groups == g)[0].tolist())
    return take


def main():
    label_map = json.load(open(LABEL_MAP_PATH)) if os.path.exists(LABEL_MAP_PATH) else None
    recs = []
    with open(os.path.join(DATA, "ge_records.csv"), newline="", encoding="utf-8",
              errors="replace") as f:
        for row in csv.DictReader(f):
            lab = row["label"].strip()
            lab = int(lab) if lab.isdigit() else label_map[lab]
            recs.append({"id": row["record_id"], "date": row.get("date", ""),
                         "customer": row.get("customer", ""),
                         "technician": row.get("technician", ""),
                         "repair": row.get("repair", ""), "label": lab,
                         "unit": row.get("unit_serial", ""),
                         "operator": row.get("operator", "")})
    print(f"{len(recs)} records")
    with gzip.open(os.path.join(DATA, "ge_records.jsonl.gz"), "wt") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    rng = np.random.default_rng(GLOBAL_SEED)
    labels = [r["label"] for r in recs]
    splits = {}
    splits["random"] = stratified_take(labels, 0.2, rng)
    from collections import Counter
    for name, grouper in [("dup_exact", exact_groups), ("dup_near", near_dup_groups)]:
        groups = grouper([r["repair"] for r in recs])
        biggest = Counter(groups).most_common(1)[0][1]
        if biggest > 0.5 * len(recs):
            print(f"split {name}: degenerate (largest group {biggest}/{len(recs)}), skipped")
            continue
        splits[name] = stratified_take(labels, 0.2, rng, groups=groups)
    if any(r["unit"] for r in recs):
        splits["unit_grouped"] = stratified_take(labels, 0.2, rng,
                                                 groups=[r["unit"] or r["id"] for r in recs])
    if any(r["operator"] for r in recs):
        splits["operator_grouped"] = stratified_take(
            labels, 0.2, rng, groups=[r["operator"] or r["id"] for r in recs])
    order = np.argsort([r["date"] for r in recs], kind="stable")
    splits["temporal"] = set(order[-round(0.2 * len(recs)):].tolist())

    out = {name: sorted(recs[i]["id"] for i in ix) for name, ix in splits.items()}
    with open(os.path.join(DATA, "splits.json"), "w") as f:
        json.dump(out, f)
    for name, ids in out.items():
        print(f"split {name}: {len(ids)} test records")


if __name__ == "__main__":
    main()
