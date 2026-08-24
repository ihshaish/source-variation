"""2x2 view matrix on the dual-report subset, balanced: BOTH models train
only on the dual-subset training records (same n, same events), one on the
primary narrative, one on the supplemental. Each is tested on both views of
the dual held-out records, plus the length-matched subset."""
import gzip,json,re,time
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
TOKEN_RE=re.compile(r"[a-z][a-z0-9/-]+"); SEED=20260802; PAD,OOV=0,1; CAP=256
recs=[json.loads(l) for l in gzip.open('views_task.jsonl.gz','rt')]
dual=[r for r in recs if len(r['r2'])>40]
print("dual records:",len(dual),"train",sum(1 for r in dual if r['split']=='train'))
device='mps' if torch.backends.mps.is_available() else 'cpu'
kv=KeyedVectors.load("w2v_narr_200d.kv",mmap='r')
def tok(t): return TOKEN_RE.findall(t.lower())[:CAP]
for r in dual:
    r['t1']=tok(r['narr']); r['t2']=tok(r['r2'])
vocab=set(t for r in dual for t in r['t1']+r['t2'])
rng=np.random.default_rng(0)
idx,vecs={},[np.zeros(200,np.float32),rng.normal(0,0.1,200).astype(np.float32)]
for w in sorted(vocab):
    if w in kv: idx[w]=len(vecs); vecs.append(kv[w].astype(np.float32))
matrix=np.stack(vecs)
def enc(key):
    X=np.zeros((len(dual),CAP),np.int32)
    for i,r in enumerate(dual):
        for j,t in enumerate(r[key]): X[i,j]=idx.get(t,OOV)
    return X
X1,X2=enc('t1'),enc('t2')
y=np.array([int(r['label']) for r in dual],np.int64)
te=np.array([r['split']=='test' for r in dual])
L1=np.array([len(r['t1']) for r in dual]); L2=np.array([len(r['t2']) for r in dual])
match=te&(L2>=0.7*L1)
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
def mf1(y,p): return f1_score(y,(p>=0.5).astype(int),average='macro')
res=[]
for train_view,Xtr_full in (('R1',X1),('R2',X2)):
    for seed in (0,1,2):
        Xtr,Xva,ytr,yva=train_test_split(Xtr_full[~te],y[~te],test_size=0.08,stratify=y[~te],random_state=SEED+seed)
        torch.manual_seed(500+seed)
        model=RNN(matrix).to(device)
        opt=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=1e-3)
        lossf=nn.CrossEntropyLoss()
        Xt=torch.from_numpy(Xtr).long(); yt=torch.from_numpy(ytr)
        best,bstate,pat=-1,None,0
        for epoch in range(20):
            model.train()
            perm=torch.randperm(len(Xt),generator=torch.Generator().manual_seed(seed*777+epoch))
            for b in range(0,len(perm),64):
                sel=perm[b:b+64]; opt.zero_grad()
                lossf(model(Xt[sel].to(device)),yt[sel].to(device)).backward(); opt.step()
            f1=mf1(yva,predict(model,Xva))
            if f1>best+1e-4: best,pat=f1,0; bstate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:
                pat+=1
                if pat>=3: break
        model.load_state_dict(bstate)
        row={"train":train_view,"seed":seed,
             "test_R1":round(float(mf1(y[te],predict(model,X1[te]))),4),
             "test_R2":round(float(mf1(y[te],predict(model,X2[te]))),4),
             "match_R1":round(float(mf1(y[match],predict(model,X1[match]))),4),
             "match_R2":round(float(mf1(y[match],predict(model,X2[match]))),4)}
        res.append(row); print(json.dumps(row),flush=True)
json.dump(res,open('matrix_2x2.json','w'),indent=1)
