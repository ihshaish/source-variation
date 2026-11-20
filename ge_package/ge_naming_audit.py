"""Naming audit with assembly_lexicon.json. For each field and class it counts
how often the text names each assembly, on the held-out events and on the
whole cohort, in the raw text (digit-initial tokens such as part numbers
kept) and in the model input after tokenising and the length cap. It also
scores a keyword rule that predicts the class from the named assemblies
alone, and for every split counts held-out events whose unit, or whose exact
text, also appears in training. Writes naming_audit.json (counts only) to
results_contrasts/, the two mask files masks/outcome_terms_v2.txt (names,
no-fault terms and replacement verbs) and masks/assembly_names_v2.txt
(names only), and the reference tokens found in the records to private/.
Run: python ge_naming_audit.py
"""
import json
import os
import re

import numpy as np

from ge_build import parse_date
from ge_lexicon import FIELDS, LEXICON, MASK_FULL, MASK_NAMES, OUT, PRIVATE, load_lexicon
from ge_lib import DATA, HERE, load_records, macro_f1, record_tokens

# unlike the model tokeniser, keeps tokens that start with a digit
RAW_RE = re.compile(r"[a-z0-9][a-z0-9/-]*")

assemblies, no_fault, verbs, class_parts = load_lexicon()
records = load_records()
splits = json.load(open(os.path.join(DATA, "splits.json")))
test_ids = set(splits["random"])
os.makedirs(os.path.dirname(MASK_FULL), exist_ok=True)
os.makedirs(OUT, exist_ok=True)
os.makedirs(PRIVATE, exist_ok=True)

# Document and part-number references are matched against the records' own
# vocabulary, so every spelling and joined form found in the text is covered.
refs = json.load(open(LEXICON)).get("references", {})
if refs:
    marker = refs["marker"].lower()
    doc_prefix = refs["document_prefix"].lower()
    vocab = set()
    for r in records:
        for field in FIELDS:
            vocab.update(RAW_RE.findall(r[field].lower()))
            vocab.update(record_tokens(r, field))
    other = set()
    for tok in vocab:
        if marker not in tok or not re.search(r"[0-9]", tok):
            continue
        hit = False
        if doc_prefix in tok:
            numbers = set(re.findall(r"[0-9]{3}", tok.split(doc_prefix, 1)[1]))
            for name, nums in refs.get("document_numbers", {}).items():
                if numbers & set(nums):
                    assemblies.setdefault(name, set()).add(tok)
                    hit = True
        for name, prefixes in refs.get("part_number_prefixes", {}).items():
            if any(tok.startswith(p.lower()) or ("/" + p.lower()) in tok for p in prefixes):
                assemblies.setdefault(name, set()).add(tok)
                hit = True
        if not hit:
            other.add(tok)
    assemblies["other_references"] = other
# the assemblies that define the classes
PARTS = set().union(*class_parts.values())

name_terms = set().union(*assemblies.values())
mask_terms = name_terms | no_fault | verbs
with open(MASK_FULL, "w") as f:
    f.write("\n".join(sorted(mask_terms)))
with open(MASK_NAMES, "w") as f:
    f.write("\n".join(sorted(name_terms)))

# the new mask should contain every term of lexicon.json, the mask of ge_leakage.py
v1_missing = []
v1_path = os.path.join(HERE, "lexicon.json")
if os.path.exists(v1_path):
    v1 = json.load(open(v1_path))
    v1_terms = {t.lower() for c in v1.get("classes", {}).values() for lst in c.values() for t in lst}
    v1_terms |= {t.lower() for t in v1.get("replacement_verbs", [])}
    v1_missing = sorted(v1_terms - mask_terms)


def raw_tokens(record, field):
    return RAW_RE.findall(record[field].lower())


def named(tokens):
    toks = set(tokens)
    return {name for name, terms in assemblies.items() if toks & terms}


def named_near_verb(tokens, window=5):
    """Assemblies named within `window` tokens of a replacement verb, a rough
    sign that the text states the replacement rather than a mention."""
    hits = set()
    verb_pos = [i for i, t in enumerate(tokens) if t in verbs]
    for i, t in enumerate(tokens):
        for name, terms in assemblies.items():
            if t in terms and any(abs(i - j) <= window for j in verb_pos):
                hits.add(name)
    return hits


def audit(rows, field):
    per_class = {}
    for c, parts in sorted(class_parts.items()):
        sub = [r for r in rows if int(r["label"]) == c]
        n = len(sub)
        summary = {"n": n}
        tallies = {}
        for r in sub:
            raw = raw_tokens(r, field)
            got_raw = named(raw)
            got_model = named(record_tokens(r, field))
            got_raw_parts = got_raw & PARTS
            near = named_near_verb(raw)
            keys = [f"raw_names_{a}" for a in got_raw]
            keys += [f"model_input_names_{a}" for a in got_model]
            keys += [f"raw_names_{a}_near_verb" for a in near]
            if got_raw & PARTS:
                keys.append("raw_any_assembly")
            if got_model & PARTS:
                keys.append("model_input_any_assembly")
            if parts and got_raw_parts & parts:
                keys.append("raw_names_any_replaced")
            if parts and parts <= got_raw_parts:
                keys.append("raw_names_all_replaced")
            if got_raw_parts - parts:
                keys.append("raw_names_assembly_not_replaced")
            if set(raw) & no_fault:
                keys.append("raw_no_fault_term")
            for k in keys:
                tallies[k] = tallies.get(k, 0) + 1
        for k, v in sorted(tallies.items()):
            summary[k] = {"count": v, "pct": round(100 * v / max(1, n), 1)}
        per_class[c] = summary
    return per_class


test_rows = [r for r in records if r["id"] in test_ids]
naming = {}
for scope, rows in (("held_out", test_rows), ("cohort", records)):
    naming[scope] = {field: audit(rows, field) for field in FIELDS}

# keyword rule on the held-out events, from the model-input tokens
train_labels = [int(r["label"]) for r in records if r["id"] not in test_ids]
majority = int(np.bincount(train_labels).argmax())
by_parts = {frozenset(p): c for c, p in class_parts.items()}
y = np.array([int(r["label"]) for r in test_rows])
keyword = {}
for field in FIELDS:
    yhat = []
    for r in test_rows:
        toks = record_tokens(r, field)
        got = named(toks) & PARTS
        if got:
            yhat.append(by_parts.get(frozenset(got), majority))
        elif set(toks) & no_fault and frozenset() in by_parts:
            yhat.append(by_parts[frozenset()])
        else:
            yhat.append(majority)
    keyword[field] = round(float(macro_f1(y, np.array(yhat))), 4)


def date_range(rows):
    dates = [parse_date(r["date"]) for r in rows]
    dates = [d for d in dates if d is not None]
    if not dates:
        return []
    return [f"{min(dates):%Y-%m-%d}", f"{max(dates):%Y-%m-%d}"]


overlap = {}
for split, ids in splits.items():
    ids = set(ids)
    train = [r for r in records if r["id"] not in ids]
    test = [r for r in records if r["id"] in ids]
    train_units = {r.get("unit") for r in train if r.get("unit")}
    with_unit = [r for r in test if r.get("unit")]
    o = {"n_test": len(test), "test_with_unit_id": len(with_unit),
         "test_units": len({r["unit"] for r in with_unit}),
         "test_events_unit_seen_in_training": sum(r["unit"] in train_units for r in with_unit)}
    for field in FIELDS:
        seen = {" ".join(r[field].lower().split()) for r in train}
        o[f"{field}_exact_copy_in_training"] = sum(" ".join(r[field].lower().split()) in seen for r in test)
    if split == "temporal":
        o["test_dates"] = date_range(test)
        o["train_dates"] = date_range(train)
        o["test_class_counts"] = np.bincount([int(r["label"]) for r in test], minlength=4).tolist()
    overlap[split] = o

out = {"lexicon_sizes": {**{k: len(v) for k, v in assemblies.items()},
                         "no_fault": len(no_fault), "replacement_verbs": len(verbs)},
       "class_parts": {c: sorted(p) for c, p in class_parts.items()},
       "v1_terms_missing_from_v2": len(v1_missing),
       "naming_by_scope_field_class": naming,
       "keyword_rule_macro_f1_held_out": keyword,
       "split_overlap": overlap}
with open(os.path.join(OUT, "naming_audit.json"), "w") as f:
    json.dump(out, f, indent=1)

with open(os.path.join(PRIVATE, "reference_tokens.txt"), "w") as f:
    for a in sorted(assemblies):
        found = sorted(t for t in assemblies[a] if re.search(r"[0-9]", t))
        f.write(f"{a}: {' '.join(found)}\n")

print("Technician field, held-out events: % naming each assembly (raw text / model input)")
for c, s in naming["held_out"]["technician"].items():
    cells = []
    for a in sorted(assemblies):
        raw = s.get(f"raw_names_{a}", {}).get("pct", 0.0)
        mod = s.get(f"model_input_names_{a}", {}).get("pct", 0.0)
        cells.append(f"{a} {raw:5.1f} / {mod:5.1f}")
    print(f"  class {c} {str(sorted(class_parts[c])):28s} n={s['n']:5d}  " + "   ".join(cells))
print("\nkeyword rule macro-F1 (held out):", keyword)
print("\nsplit overlap:")
for split, o in overlap.items():
    print(f"  {split:16s} test {o['n_test']:5d}  unit seen in training {o['test_events_unit_seen_in_training']:5d}"
          f" of {o['test_with_unit_id']}  exact copies (customer/technician/repair) "
          f"{o['customer_exact_copy_in_training']}/{o['technician_exact_copy_in_training']}/"
          f"{o['repair_exact_copy_in_training']}")
if v1_missing:
    print(f"\n{len(v1_missing)} terms of lexicon.json are missing from assembly_lexicon.json")
print(f"\nmask {MASK_FULL}: {len(mask_terms)} terms")
print(f"mask {MASK_NAMES}: {len(name_terms)} terms")
