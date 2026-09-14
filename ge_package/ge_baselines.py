"""Word- and character-n-gram TF-IDF baselines with logistic regression,
one per text field, trained on the random split and scored on its held-out
records. The character model is the comparison that copes with part
identifiers. Reads ge_records.jsonl.gz and splits.json from GE_DATA and
writes tfidf_baselines.json to GE_RES. Run: python ge_baselines.py
"""
import json
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ge_lib import DATA, RES, load_records, macro_f1

records = load_records()
split = set(json.load(open(os.path.join(DATA, "splits.json")))["random"])
train = [record for record in records if record["id"] not in split]
test = [record for record in records if record["id"] in split]
out = {}
for field in ("customer", "technician", "repair"):
    for name, options in [("word", dict(analyzer="word", ngram_range=(1, 2), min_df=2)),
                          ("char", dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2))]:
        vectoriser = TfidfVectorizer(lowercase=True, **options)
        Xtr = vectoriser.fit_transform([record[field] for record in train])
        Xte = vectoriser.transform([record[field] for record in test])
        classifier = LogisticRegression(max_iter=2000, C=1.0)
        classifier.fit(Xtr, [record["label"] for record in train])
        f1 = macro_f1(np.array([record["label"] for record in test]), classifier.predict(Xte))
        out[f"{field}_{name}"] = round(float(f1), 4)
        print(field, name, round(float(f1), 4))
json.dump(out, open(os.path.join(RES, "tfidf_baselines.json"), "w"), indent=1)
