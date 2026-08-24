"""Paired statistics for the matched-view experiment: bootstrap CIs per
configuration, paired approximate-randomisation for narrative vs synopsis
(same records, Holm within the family), and the reporter-1 vs reporter-2
paired contrast on the dual-report subset."""
import glob, json, os, numpy as np
from sklearn.metrics import f1_score
HERE=os.path.dirname(os.path.abspath(__file__))
rng=np.random.default_rng(20260802)
def mf1(y,p): return f1_score(y,(p>=0.5).astype(int),average='macro')
def boot(y,p,n=10000):
    idx=np.arange(len(y)); vals=[]
    for _ in range(n):
        s=rng.choice(idx,len(idx),replace=True); vals.append(mf1(y[s],p[s]))
    return float(np.percentile(vals,2.5)),float(np.percentile(vals,97.5))
def paired(y,pa,pb,n=10000):
    d0=mf1(y,pa)-mf1(y,pb); cnt=0
    for _ in range(n):
        sw=rng.random(len(y))<0.5
        qa=np.where(sw,pb,pa); qb=np.where(sw,pa,pb)
        if abs(mf1(y,qa)-mf1(y,qb))>=abs(d0)-1e-12: cnt+=1
    return d0,(cnt+1)/(n+1)
out={"configs":{},"contrasts":[]}
for f in sorted(glob.glob(os.path.join(HERE,"preds_*_s0.npz"))):
    key=os.path.basename(f)[6:-7]
    d=np.load(f); lo,hi=boot(d['y'],d['probs'])
    out["configs"][key]={"test_macro_f1":round(mf1(d['y'],d['probs']),4),"ci95":[round(lo,4),round(hi,4)]}
    print(key,out["configs"][key])
# narrative vs synopsis, paired by record, per embedding family, all seeds
fam=[]
for emb in ("tfidf","glove200","w2vview"):
    seeds=(0,) if emb=="tfidf" else (0,1,2)
    for s in seeds:
        fn_n=os.path.join(HERE,f"preds_narr_{emb}_s{s}.npz"); fn_s=os.path.join(HERE,f"preds_syn_{emb}_s{s}.npz")
        if not (os.path.exists(fn_n) and os.path.exists(fn_s)): continue
        dn,ds=np.load(fn_n),np.load(fn_s)
        assert list(dn['acns'])==list(ds['acns'])
        d0,p=paired(dn['y'],dn['probs'],ds['probs'])
        fam.append({"family":"view","contrast":f"{emb} s{s}: narr vs syn","delta_f1":round(d0,4),"p":round(p,4)})
# reporter 1 vs reporter 2, narrative models
for f in sorted(glob.glob(os.path.join(HERE,"dualpreds_*_s*.npz"))):
    d=np.load(f); d0,p=paired(d['y'],d['p1'],d['p2'])
    fam.append({"family":"author","contrast":os.path.basename(f)[10:-4]+": r1 vs r2","delta_f1":round(d0,4),
                "p":round(p,4),"n":int(len(d['y']))})
m=len(fam)
for i,c in enumerate(sorted(fam,key=lambda c:c["p"])):
    c["p_holm"]=round(min(1.0,c["p"]*(m-i)),4)
out["contrasts"]=fam
json.dump(out,open(os.path.join(HERE,'views_stats.json'),'w'),indent=1)
print(json.dumps(fam,indent=1))
