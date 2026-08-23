"""L1 data build — NASA ASRS Aircraft binary task per main.tex §3.3/§4.5.

Protocol (verbatim from the paper):
- 18 primary-problem categories; binary task Aircraft vs rest; blank Primary
  Problem excluded (L0 finding, 8,825 records).
- Majority-class (rest) undersampled to 50:50 against Aircraft.
- One 80/20 stratified train/test split BEFORE any preprocessing or tuning;
  global data seed fixed. Tokeniser/vocabulary statistics from train only.

Outputs l1_data/task_aircraft.jsonl.gz with {acn, year, label, text} and
l1_data/split.json with train/test ACN lists.
"""
import csv
import gzip
import json
import os
import re

import numpy as np

DATA_DIR = os.environ.get("ASRS_DIR", os.path.join(os.path.dirname(__file__), "..", "data_asrs"))
OUT = os.environ.get("L1_DATA", os.path.join(os.path.dirname(__file__), "l1_data"))
MAX_YEAR = int(os.environ.get("MAX_YEAR", "9999"))
os.makedirs(OUT, exist_ok=True)
csv.field_size_limit(10_000_000)

GLOBAL_DATA_SEED = 20260802
TOKEN_RE = re.compile(r"[a-z][a-z0-9/-]+")


def read_records(path):
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        try:
            sections = next(reader)
            fields = next(reader)
        except StopIteration:
            return []
        header = [(s.strip(), t.strip()) for s, t in zip(sections, fields)]
        idx = {}
        for key, sec, fld in [("date", "Time", "Date"),
                              ("primary", "Assessments", "Primary Problem"),
                              ("narrative", "Report 1", "Narrative")]:
            hits = [i for i, (s, t) in enumerate(header) if s == sec and t == fld]
            if not hits and key == "primary":
                hits = [i for i, (_, t) in enumerate(header) if t == "Primary Problem"]
            idx[key] = hits[0] if hits else None
        if idx["narrative"] is None or idx["date"] is None or idx["primary"] is None:
            return []
        out = []
        for row in reader:
            if len(row) <= idx["narrative"] or not row[0].strip().isdigit():
                continue
            date = row[idx["date"]].strip()
            if not re.fullmatch(r"(19|20)\d{4}", date):
                continue
            out.append((row[0].strip(), int(date) // 100,
                        row[idx["primary"]].strip(), row[idx["narrative"]].strip()))
        return out


def main():
    rng = np.random.default_rng(GLOBAL_DATA_SEED)
    seen = set()
    pos, neg = [], []
    for fn in sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv")):
        for acn, year, primary, narr in read_records(os.path.join(DATA_DIR, fn)):
            if year > MAX_YEAR:
                continue
            if acn in seen or not primary or not narr:
                continue
            seen.add(acn)
            rec = {"acn": acn, "year": year, "text": narr}
            if primary == "Aircraft":
                rec["label"] = 1
                pos.append(rec)
            else:
                rec["label"] = 0
                neg.append(rec)
    print(f"aircraft {len(pos)}, rest {len(neg)}")

    keep = rng.choice(len(neg), size=len(pos), replace=False)
    neg = [neg[i] for i in sorted(keep)]
    data = pos + neg
    order = rng.permutation(len(data))
    data = [data[i] for i in order]

    # stratified 80/20: permute within class, take 20% of each
    test_acns = set()
    for lbl in (0, 1):
        cls = [r["acn"] for r in data if r["label"] == lbl]
        cls_order = rng.permutation(len(cls))
        n_test = round(0.2 * len(cls))
        test_acns.update(cls[i] for i in cls_order[:n_test])

    with gzip.open(os.path.join(OUT, "task_aircraft.jsonl.gz"), "wt") as f:
        for r in data:
            f.write(json.dumps(r) + "\n")
    split = {
        "seed": GLOBAL_DATA_SEED,
        "n_total": len(data),
        "n_test": len(test_acns),
        "test_acns": sorted(test_acns),
    }
    with open(os.path.join(OUT, "split.json"), "w") as f:
        json.dump(split, f)
    n_tr = len(data) - len(test_acns)
    print(f"total {len(data)} (50:50), train {n_tr}, test {len(test_acns)}")

    # train-partition token stats (for the paper's data section)
    from collections import Counter
    vocab = Counter()
    lens = []
    for r in data:
        if r["acn"] in test_acns:
            continue
        toks = TOKEN_RE.findall(r["text"].lower())
        vocab.update(toks)
        lens.append(len(toks))
    lens = np.array(lens)
    stats = {
        "train_vocab_types": len(vocab),
        "train_tokens": int(lens.sum()),
        "narrative_tokens_mean": float(lens.mean()),
        "narrative_tokens_p50": int(np.percentile(lens, 50)),
        "narrative_tokens_p95": int(np.percentile(lens, 95)),
        "narrative_tokens_p99": int(np.percentile(lens, 99)),
        "pct_truncated_at_256": float((lens > 256).mean() * 100),
    }
    with open(os.path.join(OUT, "train_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
