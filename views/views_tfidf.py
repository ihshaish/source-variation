"""TF-IDF + logistic regression per view, held-out, predictions saved for
the paired tests."""
import gzip, json, os, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
HERE=os.path.dirname(os.path.abspath(__file__))
recs=[json.loads(l) for l in gzip.open(os.path.join(HERE,'views_task.jsonl.gz'),'rt')]
y=np.array([int(r['label']) for r in recs]); te=np.array([r['split']=='test' for r in recs])
acns=np.array([r['acn'] for r in recs])
for view in ('narr','syn'):
    texts=[r[view] for r in recs]
    v=TfidfVectorizer(lowercase=True, ngram_range=(1,2), min_df=3, sublinear_tf=True)
    Xtr=v.fit_transform([t for t,m in zip(texts,~te) if m])
    Xte=v.transform([t for t,m in zip(texts,te) if m])
    clf=LogisticRegression(max_iter=2000, C=1.0).fit(Xtr,y[~te])
    p=clf.predict_proba(Xte)[:,1]
    f1=f1_score(y[te],(p>=0.5).astype(int),average='macro')
    np.savez(os.path.join(HERE,f"preds_{view}_tfidf_s0.npz"),probs=p,y=y[te],acns=acns[te])
    out={"key":f"final_{view}_tfidf_s0","view":view,"emb":"tfidf","seed":0,
         "test_macro_f1":round(float(f1),4)}
    open(os.path.join(HERE,'views_results.jsonl'),'a').write(json.dumps(out)+'\n')
    print(json.dumps(out))
