"""Mean-pool MLP on both views (the paper's order-free probe), completing
the non-sequence control set for the view contrast."""
import gzip,json,re
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
TOKEN_RE=re.compile(r"[a-z][a-z0-9/-]+"); SEED=20260802; PAD,OOV=0,1
CAPS={'narr':256,'syn':64}
recs=[json.loads(l) for l in gzip.open('views_task.jsonl.gz','rt')]
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
    kv=KeyedVectors.load(f"w2v_{view}_200d.kv",mmap='r')
    rng=np.random.default_rng(0)
    idx,vecs={},[np.zeros(200,np.float32),rng.normal(0,0.1,200).astype(np.float32)]
    for w in sorted(vocab):
        if w in kv: idx[w]=len(vecs); vecs.append(kv[w].astype(np.float32))
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
        np.savez(f'preds_{view}_meanmlp_s{seed}.npz',probs=p,y=y[te])
        print(json.dumps({"view":view,"arch":"meanmlp","seed":seed,"test_macro_f1":round(float(mf1(y[te],p)),4)}),flush=True)
print("MEANPOOL DONE",flush=True)
