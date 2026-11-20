"""Word and character TF-IDF baselines per field on the random split, plain and
with masks/outcome_terms_v2.txt applied. The settings are those of
ge_baselines.py, but the text is tokenised first, so the plain and masked
inputs differ only in the masked tokens. Writes tfidf_masked.json and the
prediction arrays to results_contrasts/. Run after ge_naming_audit.py:
python ge_tfidf_masked.py
"""
import json
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ge_lexicon import FIELDS, MASK_FULL, OUT
from ge_lib import DATA, load_records, macro_f1, record_tokens

ANALYSERS = [("word", {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2}),
             ("char", {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2})]

records = load_records()
test_ids = set(json.load(open(os.path.join(DATA, "splits.json")))["random"])
mask = {line.strip() for line in open(MASK_FULL) if line.strip()}
train = [r for r in records if r["id"] not in test_ids]
test = [r for r in records if r["id"] in test_ids]
ytr = [r["label"] for r in train]
yte = np.array([r["label"] for r in test])
os.makedirs(OUT, exist_ok=True)
out = {}
for field in FIELDS:
    for masked in (False, True):
        field_mask = mask if masked else None
        text_tr = [" ".join(record_tokens(r, field, field_mask)) for r in train]
        text_te = [" ".join(record_tokens(r, field, field_mask)) for r in test]
        for name, options in ANALYSERS:
            vectoriser = TfidfVectorizer(lowercase=True, **options)
            classifier = LogisticRegression(max_iter=2000, C=1.0)
            classifier.fit(vectoriser.fit_transform(text_tr), ytr)
            pred = classifier.predict(vectoriser.transform(text_te))
            key = f"{field}_{name}"
            if masked:
                key += "_mask-outcome_terms_v2"
            out[key] = round(float(macro_f1(yte, pred)), 4)
            np.savez(os.path.join(OUT, f"preds_tfidf_{key}.npz"), yhat=pred, y=yte)
            print(key, out[key], flush=True)
with open(os.path.join(OUT, "tfidf_masked.json"), "w") as f:
    json.dump(out, f, indent=1)
