"""Corpus statistics for the ASRS export, per primary-problem category.

Reads every CSV in the export folder (ASRS_DIR), drops ACNs seen in an earlier
file, and counts records per category and year. For each category it also
reports mean narrative length, the share of de-identification placeholder
tokens (ZZZ, XX and the like), and the token and type OOV rate against the
GloVe-6B vocabulary, which is the same across the 50, 100, 200 and 300d files.
It then looks for year spans whose Aircraft count comes within 500 of the
count used in the initial study.

Writes per_task_stats.json, cat_year_counts.json and reconciliation.json to
CORPUS_OUT. Run: python3 corpus_stats.py
"""
import csv
import json
import os
import re
from collections import Counter, defaultdict

DATA_DIR = os.environ.get("ASRS_DIR", os.path.join(os.path.dirname(__file__), "..", "data_asrs"))
GLOVE = os.path.join(os.environ.get("EMB_DIR", os.path.join(os.path.dirname(__file__), "..", "embeddings")), "glove.6B.200d.txt")
OUT = os.environ.get("CORPUS_OUT", os.path.join(os.path.dirname(__file__), "corpus_out"))
os.makedirs(OUT, exist_ok=True)
csv.field_size_limit(10_000_000)

TOKEN_RE = re.compile(r"[a-z][a-z0-9/-]+")
# ASRS de-identification placeholders: ZZZ (with digit suffixes) for places,
# XX and YY forms for identifiers.
PLACEHOLDER_RE = re.compile(r"^(z{2,}\d*|x{2,}\d*|y{2,}\d*)$")


def read_records(path):
    """The export has a two-row header; returns (acn, yyyymm, primary_problem, narrative)."""
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
        if column["narrative"] is None or column["date"] is None:
            print(f"  !! no narrative/date column in {os.path.basename(path)}")
            return []
        records = []
        for row in reader:
            if len(row) <= column["narrative"] or not row[0].strip().isdigit():
                continue
            date = row[column["date"]].strip()
            if not re.fullmatch(r"(19|20)\d{4}", date):
                continue
            primary = ""
            if column["primary"] is not None and len(row) > column["primary"]:
                primary = row[column["primary"]].strip()
            records.append((row[0].strip(), int(date), primary, row[column["narrative"]].strip()))
        return records


def main():
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv"))
    print(f"{len(files)} csv files")

    seen_acn = set()
    n_dup = 0
    by_cat_year = defaultdict(Counter)
    chars = defaultdict(lambda: [0, 0])
    tok_total = defaultdict(int)
    tok_ph = defaultdict(int)
    rec_ph = defaultdict(int)
    vocab = Counter()
    missing_primary = Counter()

    for filename in files:
        records = read_records(os.path.join(DATA_DIR, filename))
        for acn, date, primary, narrative in records:
            if acn in seen_acn:
                n_dup += 1
                continue
            seen_acn.add(acn)
            year = date // 100
            cat = primary if primary else "(blank)"
            if not primary:
                missing_primary[year] += 1
            by_cat_year[cat][year] += 1
            tokens = TOKEN_RE.findall(narrative.lower())
            vocab.update(tokens)
            chars[cat][0] += len(narrative)
            chars[cat][1] += 1
            n_placeholder = sum(1 for token in tokens if PLACEHOLDER_RE.match(token))
            tok_total[cat] += len(tokens)
            tok_ph[cat] += n_placeholder
            if n_placeholder:
                rec_ph[cat] += 1
        print(f"  {filename}: +{len(records)}")

    print(f"unique ACNs {len(seen_acn)}, cross-file duplicates skipped {n_dup}")

    glove_vocab = set()
    with open(GLOVE, encoding="utf-8") as f:
        for line in f:
            glove_vocab.add(line.split(" ", 1)[0])
    print(f"glove vocab {len(glove_vocab)}")

    oov_tok = defaultdict(int)
    oov_types = defaultdict(set)
    cat_types = defaultdict(set)
    seen_acn2 = set()
    for filename in files:
        for acn, date, primary, narrative in read_records(os.path.join(DATA_DIR, filename)):
            if acn in seen_acn2:
                continue
            seen_acn2.add(acn)
            cat = primary if primary else "(blank)"
            for token in TOKEN_RE.findall(narrative.lower()):
                cat_types[cat].add(token)
                if token not in glove_vocab:
                    oov_tok[cat] += 1
                    oov_types[cat].add(token)

    rows = []
    for cat in sorted(by_cat_year, key=lambda c: -sum(by_cat_year[c].values())):
        n = sum(by_cat_year[cat].values())
        rows.append({
            "category": cat,
            "n_records": n,
            "mean_chars": round(chars[cat][0] / max(1, chars[cat][1]), 1),
            "placeholder_token_pct": round(100 * tok_ph[cat] / max(1, tok_total[cat]), 3),
            "records_with_placeholder_pct": round(100 * rec_ph[cat] / max(1, n), 1),
            "oov_token_pct_glove": round(100 * oov_tok[cat] / max(1, tok_total[cat]), 3),
            "oov_types": len(oov_types[cat]),
            "types": len(cat_types[cat]),
            "years": f"{min(by_cat_year[cat])}-{max(by_cat_year[cat])}",
        })
    with open(os.path.join(OUT, "per_task_stats.json"), "w") as f:
        json.dump(rows, f, indent=1)
    with open(os.path.join(OUT, "cat_year_counts.json"), "w") as f:
        json.dump({c: dict(y) for c, y in by_cat_year.items()}, f, indent=1)

    # 44,039 is the Aircraft count in the initial study; the export window it
    # came from is not recorded, so every year span near that count is listed
    target = 44039
    aircraft = by_cat_year.get("Aircraft", Counter())
    years = sorted(aircraft)
    spans = []
    for i in range(len(years)):
        run = 0
        for j in range(i, len(years)):
            run += aircraft[years[j]]
            if abs(run - target) <= 500:
                spans.append((years[i], years[j], run))
    aircraft_mean_chars = None
    if chars["Aircraft"][1]:
        aircraft_mean_chars = round(chars["Aircraft"][0] / max(1, chars["Aircraft"][1]), 1)
    summary = {
        "total_unique_records": len(seen_acn),
        "vocab_size_all": len(vocab),
        "thesis_targets": {"aircraft": 44039, "software": 31, "vocab": 66242,
                           "aircraft_mean_chars": 1211},
        "observed": {
            "aircraft_all_years": sum(aircraft.values()),
            "software_all_years": sum(by_cat_year.get("Software and Automation", Counter()).values()),
            "aircraft_mean_chars": aircraft_mean_chars,
        },
        "aircraft_spans_within_500_of_target": spans,
        "blank_primary_by_year": dict(missing_primary),
    }
    with open(os.path.join(OUT, "reconciliation.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
