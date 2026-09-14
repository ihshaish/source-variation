"""Builds the binary Aircraft task from the ASRS export.

Reads every CSV in ASRS_DIR, keeps one record per ACN with a non-blank
Primary Problem and a narrative, and labels Aircraft as 1 and every other
category as 0. The larger class is undersampled to a 50:50 balance, then a
stratified 80/20 train/test split is drawn once, before any preprocessing,
from a fixed data seed. Token statistics are taken from the training part
only.

Writes task_aircraft.jsonl.gz ({acn, year, label, text} per line), split.json
(the test ACNs) and train_stats.json to ASRS_DATA. Records after MAX_YEAR are
dropped. Run: python3 build_task.py
"""
import csv
import gzip
import json
import os
import re

import numpy as np

DATA_DIR = os.environ.get("ASRS_DIR", os.path.join(os.path.dirname(__file__), "..", "data_asrs"))
OUT = os.environ.get("ASRS_DATA", os.path.join(os.path.dirname(__file__), "data"))
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
        column = {}
        for key, section, field in [("date", "Time", "Date"),
                                    ("primary", "Assessments", "Primary Problem"),
                                    ("narrative", "Report 1", "Narrative")]:
            hits = [i for i, (s, t) in enumerate(header) if s == section and t == field]
            if not hits and key == "primary":
                hits = [i for i, (_, t) in enumerate(header) if t == "Primary Problem"]
            column[key] = hits[0] if hits else None
        if column["narrative"] is None or column["date"] is None or column["primary"] is None:
            return []
        records = []
        for row in reader:
            if len(row) <= column["narrative"] or not row[0].strip().isdigit():
                continue
            date = row[column["date"]].strip()
            if not re.fullmatch(r"(19|20)\d{4}", date):
                continue
            records.append((row[0].strip(), int(date) // 100,
                            row[column["primary"]].strip(), row[column["narrative"]].strip()))
        return records


def main():
    rng = np.random.default_rng(GLOBAL_DATA_SEED)
    seen = set()
    positives = []
    negatives = []
    for filename in sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv")):
        for acn, year, primary, narrative in read_records(os.path.join(DATA_DIR, filename)):
            if year > MAX_YEAR:
                continue
            if acn in seen or not primary or not narrative:
                continue
            seen.add(acn)
            record = {"acn": acn, "year": year, "text": narrative}
            if primary == "Aircraft":
                record["label"] = 1
                positives.append(record)
            else:
                record["label"] = 0
                negatives.append(record)
    print(f"aircraft {len(positives)}, rest {len(negatives)}")

    keep = rng.choice(len(negatives), size=len(positives), replace=False)
    negatives = [negatives[i] for i in sorted(keep)]
    data = positives + negatives
    order = rng.permutation(len(data))
    data = [data[i] for i in order]

    test_acns = set()
    for label in (0, 1):
        class_acns = [record["acn"] for record in data if record["label"] == label]
        class_order = rng.permutation(len(class_acns))
        n_test = round(0.2 * len(class_acns))
        test_acns.update(class_acns[i] for i in class_order[:n_test])

    with gzip.open(os.path.join(OUT, "task_aircraft.jsonl.gz"), "wt") as f:
        for record in data:
            f.write(json.dumps(record) + "\n")
    split = {
        "seed": GLOBAL_DATA_SEED,
        "n_total": len(data),
        "n_test": len(test_acns),
        "test_acns": sorted(test_acns),
    }
    with open(os.path.join(OUT, "split.json"), "w") as f:
        json.dump(split, f)
    n_train = len(data) - len(test_acns)
    print(f"total {len(data)} (50:50), train {n_train}, test {len(test_acns)}")

    from collections import Counter
    vocab = Counter()
    lengths = []
    for record in data:
        if record["acn"] in test_acns:
            continue
        tokens = TOKEN_RE.findall(record["text"].lower())
        vocab.update(tokens)
        lengths.append(len(tokens))
    lengths = np.array(lengths)
    stats = {
        "train_vocab_types": len(vocab),
        "train_tokens": int(lengths.sum()),
        "narrative_tokens_mean": float(lengths.mean()),
        "narrative_tokens_p50": int(np.percentile(lengths, 50)),
        "narrative_tokens_p95": int(np.percentile(lengths, 95)),
        "narrative_tokens_p99": int(np.percentile(lengths, 99)),
        "pct_truncated_at_256": float((lengths > 256).mean() * 100),
    }
    with open(os.path.join(OUT, "train_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
