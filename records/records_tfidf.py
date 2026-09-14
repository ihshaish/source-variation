"""Trains TF-IDF with logistic regression for each record type on the
training partition and scores the held-out set, saving the predictions
for the paired tests. Reads records_task.jsonl.gz; writes
preds_<record>_tfidf_s0.npz and appends one line per record type to
records_results.jsonl.
Run: python3 records_tfidf.py"""
import gzip
import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
records = [json.loads(line) for line in gzip.open(os.path.join(HERE, 'records_task.jsonl.gz'), 'rt')]
y = np.array([int(r['label']) for r in records])
is_test = np.array([r['split'] == 'test' for r in records])
acns = np.array([r['acn'] for r in records])
for record in ('narr', 'syn'):
    texts = [r[record] for r in records]
    vectoriser = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    Xtr = vectoriser.fit_transform([t for t, m in zip(texts, ~is_test) if m])
    Xte = vectoriser.transform([t for t, m in zip(texts, is_test) if m])
    classifier = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, y[~is_test])
    probs = classifier.predict_proba(Xte)[:, 1]
    f1 = f1_score(y[is_test], (probs >= 0.5).astype(int), average='macro')
    np.savez(os.path.join(HERE, f"preds_{record}_tfidf_s0.npz"), probs=probs, y=y[is_test], acns=acns[is_test])
    out = {"key": f"final_{record}_tfidf_s0", "view": record, "emb": "tfidf", "seed": 0,
           "test_macro_f1": round(float(f1), 4)}
    open(os.path.join(HERE, 'records_results.jsonl'), 'a').write(json.dumps(out) + '\n')
    print(json.dumps(out))
