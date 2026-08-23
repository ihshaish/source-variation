"""P1b — outcome-leakage audit + deterministic regex baseline.

Uses lexicon.json (class names, synonyms, part identifiers, replacement
verbs; the seeded file is completed with domain knowledge before running). Reports, per field and class, the share of
narratives containing each term category; runs a keyword-rule baseline
(most-specific-match wins) on the held-out set; writes masks/outcome_terms.txt
for the masked rerun via ge_train --mask-file.
"""
import json, os
import numpy as np
from ge_lib import DATA, RES, load_records, macro_f1, record_tokens

HERE = os.path.dirname(os.path.abspath(__file__))
lex = json.load(open(os.path.join(HERE, "lexicon.json")))
recs = load_records()
split = set(json.load(open(os.path.join(DATA, "splits.json")))["random"])
os.makedirs(os.path.join(HERE, "masks"), exist_ok=True)
os.makedirs(RES, exist_ok=True)

terms = {int(c): {t.lower() for cat in cats.values() for t in cat}
         for c, cats in lex["classes"].items()}
verbs = {v.lower() for v in lex["replacement_verbs"]}
all_terms = set().union(*terms.values()) | verbs
open(os.path.join(HERE, "masks", "outcome_terms.txt"), "w").write("\n".join(sorted(all_terms)))

audit = {}
for field in ("customer", "technician", "repair"):
    per = {}
    for c, tset in terms.items():
        n = hit = verb_near = 0
        for r in recs:
            if r["label"] != c:
                continue
            toks = record_tokens(r, field)
            n += 1
            pos = [i for i, t in enumerate(toks) if t in tset]
            if pos:
                hit += 1
                if any(t in verbs for i in pos for t in toks[max(0, i-4):i+5]):
                    verb_near += 1
        per[c] = {"n": n, "contains_class_terms_pct": round(100*hit/max(1,n), 1),
                  "with_replacement_verb_near_pct": round(100*verb_near/max(1,n), 1)}
    audit[field] = per

# regex/keyword baseline on held-out (rule: class with most term hits; ties/none -> majority)
test = [r for r in recs if r["id"] in split]
maj = int(np.bincount([r["label"] for r in recs if r["id"] not in split]).argmax())
base = {}
for field in ("customer", "technician", "repair"):
    yhat, y = [], []
    for r in test:
        toks = set(record_tokens(r, field))
        scores = {c: len(toks & t) for c, t in terms.items()}
        top = max(scores.values())
        yhat.append(maj if top == 0 else min(c for c, s in scores.items() if s == top))
        y.append(r["label"])
    base[field] = round(macro_f1(np.array(y), np.array(yhat)), 4)
out = {"term_audit": audit, "keyword_baseline_macro_f1": base}
json.dump(out, open(os.path.join(RES, "leakage_report.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
