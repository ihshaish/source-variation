# Feeds Table 2 (seed-averaged differences and joint record-resampling intervals) and Supplementary Tables S11 and S13. Arithmetic on the prediction files written by views_train.py, meanpool_views.py, control_roberta.py and nhtsa_leg2.py; nothing trains.
"""Seed-averaged primary contrasts with joint record-resampling CIs (Table 4),
plus matched-length threshold sensitivity (S5) and ratio-bin values for S5.
Arithmetic on stored predictions only; nothing trains."""
import gzip,json,os
import numpy as np
from sklearn.metrics import f1_score
HERE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.dirname(HERE)
VIEWS=os.path.join(ROOT,'views'); NHTSA=os.path.join(ROOT,'nhtsa'); RESULTS=os.path.join(ROOT,'results')
B=2000
mf1=lambda y,p: f1_score(y,p,average='macro')
def pred(d):
    if 'pred' in d: return np.asarray(d['pred'])
    p=d['probs']; return p.argmax(1) if p.ndim==2 else (p>=0.5).astype(int)
def ci_of(fun,y,n,seed):
    rng=np.random.default_rng(20260802+seed); bs=[]
    for _ in range(B):
        ix=rng.integers(0,n,n); bs.append(fun(ix))
    lo,hi=np.percentile(bs,[2.5,97.5]); return round(float(lo),4),round(float(hi),4)
out={}
# --- ASRS syn-narr, BiLSTM w2v ---
S=[pred(np.load(f'{HERE}/preds_syn_w2vview_s{s}.npz')) for s in (0,1,2)]
N=[pred(np.load(f'{HERE}/preds_narr_w2vview_s{s}.npz')) for s in (0,1,2)]
y=np.load(f'{HERE}/preds_syn_w2vview_s0.npz')['y']; n=len(y)
d=np.mean([mf1(y,S[r])-mf1(y,N[r]) for r in range(3)])
ci=ci_of(lambda ix: np.mean([mf1(y[ix],S[r][ix])-mf1(y[ix],N[r][ix]) for r in range(3)]),y,n,1)
out['syn_narr_bilstm']={'mean':round(float(d),4),'ci':ci}; print('syn-narr BiLSTM w2v',out['syn_narr_bilstm'])
# --- ASRS syn-narr, RoBERTa ---
S=[pred(np.load(f'{HERE}/roberta_preds_asrs_syn_s{s}.npz')) for s in (0,1,2)]
N=[pred(np.load(f'{HERE}/roberta_preds_asrs_narr_s{s}.npz')) for s in (0,1,2)]
d=np.mean([mf1(y,S[r])-mf1(y,N[r]) for r in range(3)])
ci=ci_of(lambda ix: np.mean([mf1(y[ix],S[r][ix])-mf1(y[ix],N[r][ix]) for r in range(3)]),y,n,2)
out['syn_narr_roberta']={'mean':round(float(d),4),'ci':ci}; print('syn-narr RoBERTa',out['syn_narr_roberta'])
# --- dual raw + matched subset + thresholds + ratio bins ---
recs=[json.loads(l) for l in gzip.open(f'{HERE}/views_task.jsonl.gz','rt')]
bya={r['acn']:r for r in recs if r.get('r2')}
d0=np.load(f'{HERE}/dualpreds_w2vview_s0.npz'); acns=d0['acns']; yd=d0['y']; nd=len(yd)
ratio=np.array([len(bya[a]['r2'].split())/max(1,len(bya[a]['narr'].split())) for a in acns])
P1=[( np.load(f'{HERE}/dualpreds_w2vview_s{s}.npz')['p1']>=0.5).astype(int) for s in (0,1,2)]
P2=[( np.load(f'{HERE}/dualpreds_w2vview_s{s}.npz')['p2']>=0.5).astype(int) for s in (0,1,2)]
d=np.mean([mf1(yd,P1[r])-mf1(yd,P2[r]) for r in range(3)])
ci=ci_of(lambda ix: np.mean([mf1(yd[ix],P1[r][ix])-mf1(yd[ix],P2[r][ix]) for r in range(3)]),yd,nd,3)
out['dual_raw']={'mean':round(float(d),4),'ci':ci}; print('dual raw w2v',out['dual_raw'])
m=ratio>=0.7; ym=yd[m]; nm=int(m.sum())
d=np.mean([mf1(ym,P1[r][m])-mf1(ym,P2[r][m]) for r in range(3)])
ci=ci_of(lambda ix: np.mean([mf1(ym[ix],P1[r][m][ix])-mf1(ym[ix],P2[r][m][ix]) for r in range(3)]),ym,nm,4)
out['dual_matched']={'mean':round(float(d),4),'ci':ci,'n':nm}; print('dual matched',out['dual_matched'])
# thresholds, both embeddings x 3 seeds = 6 runs
Q1={};Q2={}
for emb in ('glove200','w2vview'):
    for s in (0,1,2):
        dd=np.load(f'{HERE}/dualpreds_{emb}_s{s}.npz')
        Q1[(emb,s)]=(dd['p1']>=0.5).astype(int); Q2[(emb,s)]=(dd['p2']>=0.5).astype(int)
out['thresholds']=[]
for c in (0.5,0.6,0.7,0.8,0.9,1.0):
    mm=ratio>=c; yy=yd[mm]
    runs=[round(float(mf1(yy,Q1[k][mm])-mf1(yy,Q2[k][mm])),4) for k in Q1]
    out['thresholds'].append({'cutoff':c,'n':int(mm.sum()),'min':min(runs),'max':max(runs)})
    print(out['thresholds'][-1])
# --- interaction D, view-trained, seed-averaged (sanity per-seed first) ---
MS=[pred(np.load(f'{HERE}/preds_syn_meanmlp_s{s}.npz')) for s in (0,1,2)]
MN=[pred(np.load(f'{HERE}/preds_narr_meanmlp_s{s}.npz')) for s in (0,1,2)]
SB=[pred(np.load(f'{HERE}/preds_syn_w2vview_s{s}.npz')) for s in (0,1,2)]
NB=[pred(np.load(f'{HERE}/preds_narr_w2vview_s{s}.npz')) for s in (0,1,2)]
Dper=[round(float((mf1(y,SB[r])-mf1(y,MS[r]))-(mf1(y,NB[r])-mf1(y,MN[r]))),4) for r in range(3)]
print('D per seed sanity (expect ~{0.0195,0.0140,0.0181} in some order):',Dper)
d=np.mean(Dper)
ci=ci_of(lambda ix: np.mean([(mf1(y[ix],SB[r][ix])-mf1(y[ix],MS[r][ix]))-(mf1(y[ix],NB[r][ix])-mf1(y[ix],MN[r][ix])) for r in range(3)]),y,n,5)
out['D_viewtrained']={'mean':round(float(d),4),'ci':ci,'per_seed':Dper}; print('D avg',out['D_viewtrained'])
# --- NHTSA BiLSTM field contrasts ---
yN=np.load(f'{NHTSA}/nhtsa2_preds_summary_bilstm_s0.npz')['y']; nN=len(yN)
F={f:[pred(np.load(f'{NHTSA}/nhtsa2_preds_{f}_bilstm_s{s}.npz')) for s in (0,1,2)] for f in ('summary','conseq','remedy')}
for other,tag,seed in (('conseq','sum_conseq',6),('remedy','sum_remedy',7)):
    d=np.mean([mf1(yN,F['summary'][r])-mf1(yN,F[other][r]) for r in range(3)])
    ci=ci_of(lambda ix: np.mean([mf1(yN[ix],F['summary'][r][ix])-mf1(yN[ix],F[other][r][ix]) for r in range(3)]),yN,nN,seed)
    out[tag]={'mean':round(float(d),4),'ci':ci}; print(tag,out[tag])
json.dump(out,open(f'{RESULTS}/views/seedavg_table4.json','w'),indent=1)
print('DONE')
