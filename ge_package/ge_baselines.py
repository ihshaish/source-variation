"""Word- and character-n-gram TF-IDF baselines (logistic regression), held-out
setup; the char model is the comparator that gets on with part identifiers."""
import json, os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from ge_lib import DATA, RES, load_records, macro_f1

recs = load_records()
split = set(json.load(open(os.path.join(DATA, "splits.json")))["random"])
tr = [r for r in recs if r["id"] not in split]
te = [r for r in recs if r["id"] in split]
out = {}
for field in ("customer", "technician", "repair"):
    for name, kw in [("word", dict(analyzer="word", ngram_range=(1, 2), min_df=2)),
                     ("char", dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2))]:
        vec = TfidfVectorizer(lowercase=True, **kw)
        Xtr = vec.fit_transform([r[field] for r in tr])
        Xte = vec.transform([r[field] for r in te])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, [r["label"] for r in tr])
        f1 = macro_f1(np.array([r["label"] for r in te]), clf.predict(Xte))
        out[f"{field}_{name}"] = round(float(f1), 4)
        print(field, name, round(float(f1), 4))
json.dump(out, open(os.path.join(RES, "tfidf_baselines.json"), "w"), indent=1)
