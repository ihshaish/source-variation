"""The direct view x model interaction: D = (seq - pool | synopsis) -
(seq - pool | narrative), record-level bootstrap CI over the shared
held-out set, per same-seed w2v-family configuration. Plus: paired
TF-IDF-vs-BiLSTM per view (ranking-reversal check) and Holm over the
ensemble family."""
import json, numpy as np
from sklearn.metrics import f1_score
rng=np.random.default_rng(20260802)
def mf1(y,p): return f1_score(y,(p>=0.5).astype(int),average='macro')
def load(name): 
    d=np.load(f'/Users/Hisham/github_page/PhD_peter/views_wip/{name}')
    return d['y'],d['probs']
out={}
print("== interaction D = (seq-pool|syn) - (seq-pool|narr), w2v family:",flush=True)
res=[]
for s in (0,1,2):
    y,bs=load(f'preds_syn_w2vview_s{s}.npz'); _,ms=load(f'preds_syn_meanmlp_s{s}.npz')
    _,bn=load(f'preds_narr_w2vview_s{s}.npz'); _,mn=load(f'preds_narr_meanmlp_s{s}.npz')
    D0=(mf1(y,bs)-mf1(y,ms))-(mf1(y,bn)-mf1(y,mn))
    n=len(y); idx=np.arange(n); vals=[]
    for _ in range(5000):
        r=rng.choice(idx,n,replace=True)
        vals.append((mf1(y[r],bs[r])-mf1(y[r],ms[r]))-(mf1(y[r],bn[r])-mf1(y[r],mn[r])))
    lo,hi=np.percentile(vals,[2.5,97.5])
    res.append({"seed":s,"D":round(float(D0),4),"ci95":[round(float(lo),4),round(float(hi),4)],
                "excludes_zero":bool(lo>0 or hi<0)})
    print(res[-1],flush=True)
out['interaction']=res
print("== ranking-reversal check: TF-IDF vs BiLSTM(w2v) per view, paired:",flush=True)
def paired(y,pa,pb,n=5000):
    d0=mf1(y,pa)-mf1(y,pb); cnt=0
    for _ in range(n):
        sw=rng.random(len(y))<0.5
        if abs(mf1(y,np.where(sw,pb,pa))-mf1(y,np.where(sw,pa,pb)))>=abs(d0)-1e-12: cnt+=1
    return d0,(cnt+1)/(n+1)
rev=[]
for view in ('narr','syn'):
    yt,pt=load(f'preds_{view}_tfidf_s0.npz')
    for s in (0,1,2):
        _,pb=load(f'preds_{view}_w2vview_s{s}.npz')
        d0,p=paired(yt,pt,pb)
        rev.append({"view":view,"seed":s,"tfidf_minus_bilstm":round(float(d0),4),"p":round(float(p),4)})
        print(rev[-1],flush=True)
out['reversal']=rev
# Holm over the six ensemble contrasts (raw p from battery: recompute cleanly)
print("== ensemble family, Holm:",flush=True)
ens=[]
for emb in ('glove200','w2vview'):
    for s in (0,1,2):
        y,pn=load(f'preds_narr_{emb}_s{s}.npz'); _,ps=load(f'preds_syn_{emb}_s{s}.npz')
        pe=(pn+ps)/2
        best=ps if mf1(y,ps)>=mf1(y,pn) else pn
        d0,p=paired(y,pe,best)
        ens.append({"emb":emb,"seed":s,"delta":round(float(d0),4),"p":round(float(p),4)})
m=len(ens)
for i,c in enumerate(sorted(ens,key=lambda c:c["p"])): c["p_holm"]=round(min(1.0,c["p"]*(m-i)),4)
for c in ens: print(c,flush=True)
out['ensemble_holm']=ens
json.dump(out,open('/Users/Hisham/github_page/PhD_peter/views_wip/interaction_test.json','w'),indent=1)
print("INTERACTION DONE",flush=True)
