"""Review control: the interaction D under one SHARED representation.
Trains the mean-pool MLP on both views with GloVe-200 (the fixed embedding
common to both views), then recomputes D = (seq - pool | syn) - (seq - pool
| narr) pairing the existing BiLSTM/GloVe-200 predictions with these
mean-pool runs. Training protocol identical to meanpool_views.py; only the
embedding matrix changes. Post-hoc control requested at review; interpretation
pre-committed: CI excluding zero in all seeds leaves Claim 6 unchanged and
representation-independent; otherwise the claim is scoped to view pipelines."""
import gzip,json,re
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
HERE='/Users/Hisham/github_page/PhD_peter/views_wip'
GLOVE='/Users/Hisham/github_page/PhD_peter/embeddings/glove.6B.200d.txt'
TOKEN_RE=re.compile(r"[a-z][a-z0-9/-]+"); SEED=20260802; PAD,OOV=0,1
CAPS={'narr':256,'syn':64}
recs=[json.loads(l) for l in gzip.open(f'{HERE}/views_task.jsonl.gz','rt')]
device='mps' if torch.backends.mps.is_available() else 'cpu'
class MeanMLP(nn.Module):
    def __init__(s,m,hidden=128):
        super().__init__()
        s.emb=nn.Embedding.from_pretrained(torch.from_numpy(m),freeze=True,padding_idx=PAD)
        s.fc1=nn.Linear(m.shape[1],hidden); s.drop=nn.Dropout(0.3); s.fc2=nn.Linear(hidden,2)
    def forward(s,x):
        e=s.emb(x); mask=(x!=PAD).unsqueeze(2).float()
        m=(e*mask).sum(1)/mask.sum(1).clamp(min=1.0)
        return s.fc2(s.drop(torch.relu(s.fc1(m))))
def predict(model,X):
    model.eval(); outs=[]
    with torch.no_grad():
        for b in range(0,len(X),512):
            outs.append(torch.softmax(model(torch.from_numpy(X[b:b+512]).long().to(device)),1)[:,1].cpu().numpy())
    return np.concatenate(outs)
def mf1(y,p): return f1_score(y,(p>=0.5).astype(int),average='macro')

for view in ('syn','narr'):
    cap=CAPS[view]
    toks=[TOKEN_RE.findall(r[view].lower())[:cap] for r in recs]
    vocab=set(t for ts in toks for t in ts)
    rng=np.random.default_rng(0)
    idx,vecs={},[np.zeros(200,np.float32),rng.normal(0,0.1,200).astype(np.float32)]
    for line in open(GLOVE,encoding='utf-8'):
        p=line.rstrip().split(' ')
        if p[0] in vocab: idx[p[0]]=len(vecs); vecs.append(np.asarray(p[1:],np.float32))
    matrix=np.stack(vecs)
    X=np.zeros((len(toks),cap),np.int32)
    for i,ts in enumerate(toks):
        for j,t in enumerate(ts): X[i,j]=idx.get(t,OOV)
    y=np.array([int(r['label']) for r in recs],np.int64)
    te=np.array([r['split']=='test' for r in recs])
    for seed in (0,1,2):
        Xtr,Xva,ytr,yva=train_test_split(X[~te],y[~te],test_size=0.05,stratify=y[~te],random_state=SEED+seed)
        torch.manual_seed(300+seed)
        model=MeanMLP(matrix).to(device)
        opt=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=1e-3)
        lossf=nn.CrossEntropyLoss()
        Xt=torch.from_numpy(Xtr).long(); yt=torch.from_numpy(ytr)
        best,bstate,pat=-1,None,0
        for epoch in range(15):
            model.train()
            perm=torch.randperm(len(Xt),generator=torch.Generator().manual_seed(seed*991+epoch))
            for b in range(0,len(perm),128):
                sel=perm[b:b+128]; opt.zero_grad()
                lossf(model(Xt[sel].to(device)),yt[sel].to(device)).backward(); opt.step()
            f1=mf1(yva,predict(model,Xva))
            if f1>best+1e-4: best,pat=f1,0; bstate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:
                pat+=1
                if pat>=2: break
        model.load_state_dict(bstate)
        p=predict(model,X[te])
        np.savez(f'{HERE}/preds_{view}_meanmlpglove_s{seed}.npz',probs=p,y=y[te])
        print(json.dumps({"view":view,"arch":"meanmlp-glove","seed":seed,
                          "test_macro_f1":round(float(mf1(y[te],p)),4)}),flush=True)

# D under the shared representation
rng=np.random.default_rng(20260802)
def load(name):
    d=np.load(f'{HERE}/{name}')
    return d['y'],d['probs']
res=[]
for s in (0,1,2):
    y,bs=load(f'preds_syn_glove200_s{s}.npz'); _,ms=load(f'preds_syn_meanmlpglove_s{s}.npz')
    _,bn=load(f'preds_narr_glove200_s{s}.npz'); _,mn=load(f'preds_narr_meanmlpglove_s{s}.npz')
    D0=(mf1(y,bs)-mf1(y,ms))-(mf1(y,bn)-mf1(y,mn))
    n=len(y); idx2=np.arange(n); vals=[]
    for _ in range(5000):
        r=rng.choice(idx2,n,replace=True)
        vals.append((mf1(y[r],bs[r])-mf1(y[r],ms[r]))-(mf1(y[r],bn[r])-mf1(y[r],mn[r])))
    lo,hi=np.percentile(vals,[2.5,97.5])
    res.append({"seed":s,"D":round(float(D0),4),"ci95":[round(float(lo),4),round(float(hi),4)],
                "excludes_zero":bool(lo>0 or hi<0)})
    print(res[-1],flush=True)
json.dump({"interaction_glove":res},open(f'{HERE}/control_glove_D.json','w'),indent=1)
print("GLOVE_D DONE",flush=True)
