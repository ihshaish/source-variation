"""L0 — corpus reconciliation and per-task statistics for Paper A's NASA side.

Rebuilds the 18 primary-problem categories from the raw DBOL exports and checks
the corpus facts quoted in main.tex §3.3 (Aircraft 44,039; Software and
Automation 31; vocabulary 66,242; mean Aircraft narrative 1,211 chars).
Reconciliation is span-aware: counts are kept per year so the thesis export
window can be identified by matching, not assumed.

Also computes, per category: placeholder-token density (the de-identification
tokens ZZZ/ZZZZ etc.), narrative length, and token/record-level OOV under each
GloVe-6B dimensionality (vocabulary identical across 50/100/200/300d, so one
vocab file suffices).

Usage: python3 l0_corpus_stats.py  (from paper_a_nasa/; writes l0_out/)
"""
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

DATA_DIR = os.environ.get("ASRS_DIR", os.path.join(os.path.dirname(__file__), "..", "data_asrs"))
GLOVE = os.path.join(os.path.dirname(__file__), "..", "embeddings", "glove.6B.200d.txt")
OUT = os.environ.get("L0_OUT", os.path.join(os.path.dirname(__file__), "l0_out"))
os.makedirs(OUT, exist_ok=True)
csv.field_size_limit(10_000_000)

TOKEN_RE = re.compile(r"[a-z][a-z0-9/-]+")
# ASRS de-identification placeholders: ZZZ (+digit suffixes) for locations,
# XX*/YY* variants for identifiers.
PLACEHOLDER_RE = re.compile(r"^(z{2,}\d*|x{2,}\d*|y{2,}\d*)$")


def read_records(path):
    """Two-row DBOL header; returns (acn, yyyymm, primary_problem, narrative)."""
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
        if idx["narrative"] is None or idx["date"] is None:
            print(f"  !! no narrative/date column in {os.path.basename(path)}")
            return []
        out = []
        for row in reader:
            if len(row) <= idx["narrative"] or not row[0].strip().isdigit():
                continue
            date = row[idx["date"]].strip()
            if not re.fullmatch(r"(19|20)\d{4}", date):
                continue
            primary = row[idx["primary"]].strip() if idx["primary"] is not None and len(row) > idx["primary"] else ""
            out.append((row[0].strip(), int(date), primary, row[idx["narrative"]].strip()))
        return out


def main():
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv"))
    print(f"{len(files)} csv files")

    seen_acn = set()
    n_dup = 0
    by_cat_year = defaultdict(Counter)          # cat -> year -> n
    chars = defaultdict(lambda: [0, 0])         # cat -> [sum_chars, n]
    tok_total = defaultdict(int)                # cat -> tokens
    tok_ph = defaultdict(int)                   # cat -> placeholder tokens
    rec_ph = defaultdict(int)                   # cat -> records with >=1 placeholder
    vocab = Counter()
    missing_primary = Counter()

    for fn in files:
        recs = read_records(os.path.join(DATA_DIR, fn))
        for acn, date, primary, narr in recs:
            if acn in seen_acn:
                n_dup += 1
                continue
            seen_acn.add(acn)
            year = date // 100
            cat = primary if primary else "(blank)"
            if not primary:
                missing_primary[year] += 1
            by_cat_year[cat][year] += 1
            toks = TOKEN_RE.findall(narr.lower())
            vocab.update(toks)
            chars[cat][0] += len(narr)
            chars[cat][1] += 1
            ph = sum(1 for t in toks if PLACEHOLDER_RE.match(t))
            tok_total[cat] += len(toks)
            tok_ph[cat] += ph
            if ph:
                rec_ph[cat] += 1
        print(f"  {fn}: +{len(recs)}")

    print(f"unique ACNs {len(seen_acn)}, cross-file duplicates skipped {n_dup}")

    glove_vocab = set()
    with open(GLOVE, encoding="utf-8") as f:
        for line in f:
            glove_vocab.add(line.split(" ", 1)[0])
    print(f"glove vocab {len(glove_vocab)}")

    # second pass for OOV needs tokens again; cheaper to compute from vocab counter
    # per-category OOV requires per-category token counts vs glove: redo per cat
    oov_tok = defaultdict(int)
    oov_types = defaultdict(set)
    cat_types = defaultdict(set)
    seen_acn2 = set()
    for fn in files:
        for acn, date, primary, narr in read_records(os.path.join(DATA_DIR, fn)):
            if acn in seen_acn2:
                continue
            seen_acn2.add(acn)
            cat = primary if primary else "(blank)"
            for t in TOKEN_RE.findall(narr.lower()):
                cat_types[cat].add(t)
                if t not in glove_vocab:
                    oov_tok[cat] += 1
                    oov_types[cat].add(t)

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

    # reconciliation: find year-spans matching the thesis Aircraft count
    tgt = 44039
    ac = by_cat_year.get("Aircraft", Counter())
    years = sorted(ac)
    spans = []
    for i in range(len(years)):
        run = 0
        for j in range(i, len(years)):
            run += ac[years[j]]
            if abs(run - tgt) <= 500:
                spans.append((years[i], years[j], run))
    summary = {
        "total_unique_records": len(seen_acn),
        "vocab_size_all": len(vocab),
        "thesis_targets": {"aircraft": 44039, "software": 31, "vocab": 66242,
                           "aircraft_mean_chars": 1211},
        "observed": {
            "aircraft_all_years": sum(ac.values()),
            "software_all_years": sum(by_cat_year.get("Software and Automation", Counter()).values()),
            "aircraft_mean_chars": round(chars["Aircraft"][0] / max(1, chars["Aircraft"][1]), 1)
            if chars["Aircraft"][1] else None,
        },
        "aircraft_spans_within_500_of_target": spans,
        "blank_primary_by_year": dict(missing_primary),
    }
    with open(os.path.join(OUT, "reconciliation.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
