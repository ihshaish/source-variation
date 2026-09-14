"""Outcome-leakage audit and keyword baseline. Reads lexicon.json (class
names, synonyms, part identifiers and replacement verbs; the seed file is
completed with domain knowledge before running) and reports, per field and
class, the share of records containing a class term and the share with a
replacement verb within four tokens of one. Runs a keyword-rule baseline on
the held-out records of the random split, where the class with the most term
hits wins. Writes leakage_report.json to GE_RES and masks/outcome_terms.txt,
the mask file for the masked rerun with ge_train.py --mask-file.
Run: python ge_leakage.py
"""
import json
import os

import numpy as np

from ge_lib import DATA, RES, load_records, macro_f1, record_tokens

HERE = os.path.dirname(os.path.abspath(__file__))
lexicon = json.load(open(os.path.join(HERE, "lexicon.json")))
records = load_records()
split = set(json.load(open(os.path.join(DATA, "splits.json")))["random"])
os.makedirs(os.path.join(HERE, "masks"), exist_ok=True)
os.makedirs(RES, exist_ok=True)

terms = {int(label): {term.lower() for category in categories.values() for term in category}
         for label, categories in lexicon["classes"].items()}
verbs = {verb.lower() for verb in lexicon["replacement_verbs"]}
all_terms = set().union(*terms.values()) | verbs
open(os.path.join(HERE, "masks", "outcome_terms.txt"), "w").write("\n".join(sorted(all_terms)))

audit = {}
for field in ("customer", "technician", "repair"):
    per_class = {}
    for label, term_set in terms.items():
        count = 0
        hit = 0
        verb_near = 0
        for record in records:
            if record["label"] != label:
                continue
            tokens = record_tokens(record, field)
            count += 1
            positions = [i for i, token in enumerate(tokens) if token in term_set]
            if positions:
                hit += 1
                if any(token in verbs for i in positions
                       for token in tokens[max(0, i - 4):i + 5]):
                    verb_near += 1
        per_class[label] = {"n": count,
                            "contains_class_terms_pct": round(100 * hit / max(1, count), 1),
                            "with_replacement_verb_near_pct":
                                round(100 * verb_near / max(1, count), 1)}
    audit[field] = per_class

# no term hits: training majority class; tie on hits: lowest class index
test = [record for record in records if record["id"] in split]
train_labels = [record["label"] for record in records if record["id"] not in split]
majority = int(np.bincount(train_labels).argmax())
baseline = {}
for field in ("customer", "technician", "repair"):
    yhat = []
    y = []
    for record in test:
        tokens = set(record_tokens(record, field))
        scores = {label: len(tokens & term_set) for label, term_set in terms.items()}
        top = max(scores.values())
        if top == 0:
            yhat.append(majority)
        else:
            yhat.append(min(label for label, score in scores.items() if score == top))
        y.append(record["label"])
    baseline[field] = round(macro_f1(np.array(y), np.array(yhat)), 4)
out = {"term_audit": audit, "keyword_baseline_macro_f1": baseline}
json.dump(out, open(os.path.join(RES, "leakage_report.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
