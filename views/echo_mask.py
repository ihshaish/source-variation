"""Category-echo diagnostic. Mask rule: tokens appearing in ASRS category
labels themselves (taxonomy-speak), applied at evaluation to BOTH views
symmetrically. If the synopsis advantage rides analyst taxonomy phrasing,
masking should cost the synopsis far more than the narrative."""
import gzip,json,re,time
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors

ECHO={'acft','aircraft','equip','equipment','prob','problem','malfunction',
      'malfunctioning','critical','nmac','atc','wx','weather'}
TOKEN_RE=re.compile(r"[a-z][a-z0-9/-]+"); SEED=20260802; PAD,OOV=0,1
CAPS={'narr':256,'syn':64}
recs=[json.loads(l) for l in gzip.open('views_task.jsonl.gz','rt')]
device='mps' if torch.backends.mps.is_available() else 'cpu'

class RNN(nn.Module):
    def __init__(s,m):
        super().__init__()
        s.emb=nn.Embedding.from_pretrained(torch.from_numpy(m),freeze=True,padding_idx=PAD)
        s.rnn=nn.LSTM(m.shape[1],64,batch_first=True,bidirectional=True)
        s.drop=nn.Dropout(0.3); s.fc=nn.Linear(128,2)
    def forward(s,x):
        h=s.rnn(s.emb(x))[1][0]
        return s.fc(s.drop(torch.cat([h[0],h[1]],dim=1)))

def predict(model,X):
    model.eval(); outs=[]
    with torch.no_grad():
        for b in range(0,len(X),256):
            outs.append(torch.softmax(model(torch.from_numpy(X[b:b+256]).long().to(device)),1)[:,1].cpu().numpy())
    return np.concatenate(outs)

out={}
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
    def enc(tl,mask=False):
        X=np.zeros((len(tl),cap),np.int32)
        for i,ts in enumerate(tl):
            for j,t in enumerate(ts):
                X[i,j]=OOV if (mask and t in ECHO) else idx.get(t,OOV)
        return X
    y=np.array([int(r['label']) for r in recs],np.int64)
    te=np.array([r['split']=='test' for r in recs])
    X=enc(toks); Xm=enc(toks,mask=True)
    # echo prevalence
    hit=np.mean([any(t in ECHO for t in ts) for ts,m in zip(toks,te) if m])
    Xtr,Xva,ytr,yva=train_test_split(X[~te],y[~te],test_size=0.05,stratify=y[~te],random_state=SEED)
    torch.manual_seed(100)
    model=RNN(matrix).to(device)
    opt=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=1e-3)
    lossf=nn.CrossEntropyLoss()
    Xt=torch.from_numpy(Xtr).long(); yt=torch.from_numpy(ytr)
    best,bstate,pat=-1,None,0
    for epoch in range(15):
        model.train()
        perm=torch.randperm(len(Xt),generator=torch.Generator().manual_seed(100000+epoch))
        for b in range(0,len(perm),128):
            sel=perm[b:b+128]; opt.zero_grad()
            lossf(model(Xt[sel].to(device)),yt[sel].to(device)).backward(); opt.step()
        f1=f1_score(yva,(predict(model,Xva)>=0.5).astype(int),average='macro')
        if f1>best+1e-4: best,pat=f1,0; bstate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            pat+=1
            if pat>=2: break
    model.load_state_dict(bstate)
    f_plain=f1_score(y[te],(predict(model,X[te])>=0.5).astype(int),average='macro')
    f_mask=f1_score(y[te],(predict(model,Xm[te])>=0.5).astype(int),average='macro')
    out[view]={"plain":round(float(f_plain),4),"echo_masked":round(float(f_mask),4),
               "cost":round(float(f_plain-f_mask),4),"echo_hit_rate":round(float(hit),3)}
    print(view,out[view])
json.dump(out,open('echo_mask.json','w'),indent=1)
