# Feeds Supplementary Table S20: the three RoBERTa contrasts declared before the ASRS runs completed (D against BiLSTM and TF-IDF, NHTSA gains, summary TF-IDF vs RoBERTa).
"""Pre-declared RoBERTa contrasts 2-4 (CLAIMS_REGISTER_v5.1_controls.md,
plan of 2026-08-25, declared before the ASRS runs completed).
2. D_R = [S(syn,RoBERTa)-S(syn,BiLSTM_view)] - [S(narr,RoBERTa)-S(narr,BiLSTM_view)],
   record bootstrap per seed; secondary baseline TF-IDF (s0 preds).
3. NHTSA gains G_v = S(v,RoBERTa)-S(v,BiLSTM_field); G_remedy vs G_summary,
   record bootstrap of the difference, per seed.
4. NHTSA summary: TF-IDF vs RoBERTa paired permutation test, per seed.
Emits control_roberta_prereg.json."""
import json,os
import numpy as np
from sklearn.metrics import f1_score
HERE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.dirname(HERE)
VIEWS=os.path.join(ROOT,'views'); NHTSA=os.path.join(ROOT,'nhtsa'); RESULTS=os.path.join(ROOT,'results')
B=2000
def mf1(yy,p): return f1_score(yy,p,average='macro')
def pred(d):
    if 'pred' in d: return d['pred']
    p=d['probs']
    return p.argmax(1) if p.ndim==2 else (p>=0.5).astype(int)
def paired(yy,pa,pb,rng,n=5000):
    d0=mf1(yy,pa)-mf1(yy,pb); cnt=0
    for _ in range(n):
        sw=rng.random(len(yy))<0.5
        if abs(mf1(yy,np.where(sw,pb,pa))-mf1(yy,np.where(sw,pa,pb)))>=abs(d0)-1e-12: cnt+=1
    return round(float(d0),4),round((cnt+1)/(n+1),4)
out={}
# ---- contrast 2: D_R on ASRS ------------------------------------------------
out['DR']=[]
for s in (0,1,2):
    sr=np.load(os.path.join(HERE,f'roberta_preds_asrs_syn_s{s}.npz'))
    nr=np.load(os.path.join(HERE,f'roberta_preds_asrs_narr_s{s}.npz'))
    y=sr['y']
    for base,tag in ((f'w2vview_s{s}','bilstm'),('tfidf_s0','tfidf')):
        sb=np.load(os.path.join(HERE,f'preds_syn_{base}.npz'))
        nb=np.load(os.path.join(HERE,f'preds_narr_{base}.npz'))
        ps_r,pn_r,ps_b,pn_b=pred(sr),pred(nr),pred(sb),pred(nb)
        d0=(mf1(y,ps_r)-mf1(y,ps_b))-(mf1(y,pn_r)-mf1(y,pn_b))
        rng=np.random.default_rng(20260802+s)
        bs=[]
        for _ in range(B):
            ix=rng.integers(0,len(y),len(y)); yy=y[ix]
            bs.append((mf1(yy,ps_r[ix])-mf1(yy,ps_b[ix]))-(mf1(yy,pn_r[ix])-mf1(yy,pn_b[ix])))
        lo,hi=np.percentile(bs,[2.5,97.5])
        out['DR'].append({"seed":s,"baseline":tag,"D_R":round(float(d0),4),
                          "ci":[round(float(lo),4),round(float(hi),4)]})
        print(out['DR'][-1],flush=True)
# ---- contrast 3: NHTSA gain comparison --------------------------------------
out['nhtsa_gains']=[]
for s in (0,1,2):
    rb={v:pred(np.load(os.path.join(HERE,f'roberta_preds_nhtsa_{v}_s{s}.npz'))) for v in ('summary','remedy')}
    bl={v:pred(np.load(os.path.join(NHTSA,f'nhtsa2_preds_{v}_bilstm_s{s}.npz'))) for v in ('summary','remedy')}
    y=np.load(os.path.join(HERE,f'roberta_preds_nhtsa_summary_s{s}.npz'))['y']
    d0=(mf1(y,rb['remedy'])-mf1(y,bl['remedy']))-(mf1(y,rb['summary'])-mf1(y,bl['summary']))
    rng=np.random.default_rng(20260802+s)
    bs=[]
    for _ in range(B):
        ix=rng.integers(0,len(y),len(y)); yy=y[ix]
        bs.append((mf1(yy,rb['remedy'][ix])-mf1(yy,bl['remedy'][ix]))-(mf1(yy,rb['summary'][ix])-mf1(yy,bl['summary'][ix])))
    lo,hi=np.percentile(bs,[2.5,97.5])
    Gs=round(float(mf1(y,rb['summary'])-mf1(y,bl['summary'])),4)
    Gr=round(float(mf1(y,rb['remedy'])-mf1(y,bl['remedy'])),4)
    out['nhtsa_gains'].append({"seed":s,"G_summary":Gs,"G_remedy":Gr,
        "G_remedy_minus_G_summary":round(float(d0),4),
        "ci":[round(float(lo),4),round(float(hi),4)]})
    print(out['nhtsa_gains'][-1],flush=True)
# ---- contrast 4: NHTSA summary TF-IDF vs RoBERTa ----------------------------
out['nhtsa_summary_tfidf_vs_roberta']=[]
tf=np.load(os.path.join(NHTSA,'nhtsa2_preds_summary_tfidf_s0.npz'))
for s in (0,1,2):
    rb=np.load(os.path.join(HERE,f'roberta_preds_nhtsa_summary_s{s}.npz'))
    rng=np.random.default_rng(20260802+s)
    d,p=paired(rb['y'],pred(tf),pred(rb),rng)
    out['nhtsa_summary_tfidf_vs_roberta'].append(
        {"seed":s,"delta_tfidf_minus_roberta":d,"p":p})
    print(out['nhtsa_summary_tfidf_vs_roberta'][-1],flush=True)
json.dump(out,open(os.path.join(RESULTS,'views','control_roberta_prereg.json'),'w'),indent=1)
print("PREREG DONE",flush=True)
