"""Character n-gram baseline per record: TF-IDF over character 3- to 5-grams
within word boundaries and logistic regression, trained on the training
partition of the narrative and of the synopsis and scored on the held-out
set, with a paired approximate-randomisation test between the two records.
Reads records_task.jsonl.gz; writes preds_narr_charngram_s0.npz,
preds_syn_charngram_s0.npz and records_charngram.json.

    python3 records_charngram.py
"""
import gzip
import json
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(20260802)

records = [json.loads(line) for line in gzip.open(os.path.join(HERE, "records_task.jsonl.gz"), "rt")]
y = np.array([int(record["label"]) for record in records])
is_test = np.array([record["split"] == "test" for record in records])


def macro_f1(labels, probs):
    return f1_score(labels, (probs >= 0.5).astype(int), average="macro")


def paired_test(labels, probs_a, probs_b, n_permutations=5000):
    observed = macro_f1(labels, probs_a) - macro_f1(labels, probs_b)
    count = 0
    for _ in range(n_permutations):
        swap = rng.random(len(labels)) < 0.5
        permuted = macro_f1(labels, np.where(swap, probs_b, probs_a)) - macro_f1(labels, np.where(swap, probs_a, probs_b))
        if abs(permuted) >= abs(observed) - 1e-12:
            count += 1
    return observed, (count + 1) / (n_permutations + 1)


result = {}
for record_key in ("narr", "syn"):
    vectoriser = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, sublinear_tf=True, max_features=300000)
    X_train = vectoriser.fit_transform([record[record_key] for record, keep in zip(records, ~is_test) if keep])
    X_test = vectoriser.transform([record[record_key] for record, keep in zip(records, is_test) if keep])
    classifier = LogisticRegression(max_iter=1500).fit(X_train, y[~is_test])
    probs = classifier.predict_proba(X_test)[:, 1]
    result[record_key] = round(float(macro_f1(y[is_test], probs)), 4)
    print(record_key, result[record_key], flush=True)
    np.savez(os.path.join(HERE, f"preds_{record_key}_charngram_s0.npz"), probs=probs, y=y[is_test])

narrative = np.load(os.path.join(HERE, "preds_narr_charngram_s0.npz"))
synopsis = np.load(os.path.join(HERE, "preds_syn_charngram_s0.npz"))
delta, p = paired_test(narrative["y"], narrative["probs"], synopsis["probs"])
result["narr_minus_syn"] = round(float(delta), 4)
result["p"] = round(p, 4)
print("narrative minus synopsis", result["narr_minus_syn"], "p", result["p"])
json.dump(result, open(os.path.join(HERE, "records_charngram.json"), "w"), indent=1)
