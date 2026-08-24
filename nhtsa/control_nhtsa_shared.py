"""Review control: the NHTSA field hierarchy under one SHARED representation.
Identical to nhtsa_leg2.py's BiLSTM leg except that a single word2vec model
is trained once on the concatenation of all three fields' training text,
then frozen and reused for summary, consequence and remedy. Field changes;
embedding, model, targets and split stay fixed. Post-hoc control requested
at review; interpretation pre-committed: hierarchy persisting Holm-significant
leaves Claim 3 unchanged under the strictest field-only comparison; otherwise
the field contrast is scoped to fields-with-their-fitted-representations."""
import json,os,re
import numpy as np
HERE='/Users/Hisham/github_page/PhD_peter/views_wip'
TOKEN_RE=re.compile(r"[a-z][a-z0-9/-]+"); SEED=20260802
def log(*a): print(*a,flush=True)

rows=[json.loads(l) for l in open(os.path.join(HERE,'nhtsa_campaigns.jsonl'))]
seen=set(); camps=[]
for r in rows:
    if r['NHTSACampaignNumber'] in seen: continue
    seen.add(r['NHTSACampaignNumber']); camps.append(r)
from collections import Counter
def top(c): return (c or '').split(':')[0].split(',')[0].strip()
cnt=Counter(top(r['Component']) for r in camps)
classes=sorted([k for k,v in cnt.items() if v>=300 and k])
lab={k:i for i,k in enumerate(classes)}
data=[r for r in camps if top(r['Component']) in lab]
rng=np.random.default_rng(SEED)
parent=list(range(len(data)))
def find(i):
    while parent[i]!=i: parent[i]=parent[parent[i]]; i=parent[i]
    return i
def union(a,b):
    ra,rb=find(a),find(b)
    if ra!=rb: parent[rb]=ra
for field in ('Summary','Consequence','Remedy'):
    first={}
    for i,r in enumerate(data):
        k=(r[field] or '').strip().lower()[:400]
        if not k: continue
        if k in first: union(first[k],i)
        else: first[k]=i
groups={}
for i,r in enumerate(data): groups.setdefault(find(i),[]).append(r)
gkeys=list(groups); rng.shuffle(gkeys)
ntest=int(0.2*len(data)); test=set(); c=0
for k in gkeys:
    if c>=ntest: break
    for r in groups[k]: test.add(r['NHTSACampaignNumber'])
    c+=len(groups[k])
for r in data: r['split']='test' if r['NHTSACampaignNumber'] in test else 'train'
y=np.array([lab[top(r['Component'])] for r in data]); te=np.array([r['split']=='test' for r in data])
log("task:",len(data),"test:",int(te.sum()),"classes:",len(classes))

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
def mf1(yy,p): return f1_score(yy,p,average='macro')
VIEWS={'summary':'Summary','conseq':'Consequence','remedy':'Remedy'}
CAP=96; PAD,OOV=0,1

# ONE word2vec over the concatenation of all three fields' training text
from gensim.models import Word2Vec
toks_by={vk:[TOKEN_RE.findall((r[f] or '').lower())[:CAP] for r in data] for vk,f in VIEWS.items()}
shared_sents=[t for vk in VIEWS for t,m in zip(toks_by[vk],~te) if m and t]
w2v=Word2Vec(vector_size=200,window=5,min_count=3,sg=1,epochs=10,workers=8,seed=SEED)
w2v.build_vocab(shared_sents); w2v.train(corpus_iterable=shared_sents,total_examples=len(shared_sents),epochs=10)
vocab=set(t for vk in VIEWS for ts in toks_by[vk] for t in ts)
rng2=np.random.default_rng(0)
idx,vecs={},[np.zeros(200,np.float32),rng2.normal(0,0.1,200).astype(np.float32)]
for w in sorted(vocab):
    if w in w2v.wv: idx[w]=len(vecs); vecs.append(w2v.wv[w].astype(np.float32))
matrix=np.stack(vecs)
log("shared w2v vocab:",len(idx))

import torch, torch.nn as nn
device='mps' if torch.backends.mps.is_available() else 'cpu'
class RNN(nn.Module):
    def __init__(s,m,nc):
        super().__init__()
        s.emb=nn.Embedding.from_pretrained(torch.from_numpy(m),freeze=True,padding_idx=PAD)
        s.rnn=nn.LSTM(m.shape[1],64,batch_first=True,bidirectional=True)
        s.drop=nn.Dropout(0.3); s.fc=nn.Linear(128,nc)
    def forward(s,x):
        h=s.rnn(s.emb(x))[1][0]
        return s.fc(s.drop(torch.cat([h[0],h[1]],dim=1)))
def predict(model,X):
    model.eval(); outs=[]
    with torch.no_grad():
        for b in range(0,len(X),512):
            outs.append(model(torch.from_numpy(X[b:b+512]).long().to(device)).argmax(1).cpu().numpy())
    return np.concatenate(outs)

res=[]; preds={}
for vk in VIEWS:
    toks=toks_by[vk]
    X=np.zeros((len(toks),CAP),np.int32)
    for i,ts in enumerate(toks):
        for j,t in enumerate(ts): X[i,j]=idx.get(t,OOV)
    for seed in (0,1,2):
        Xtr,Xva,ytr,yva=train_test_split(X[~te],y[~te],test_size=0.05,stratify=y[~te],random_state=SEED+seed)
        torch.manual_seed(700+seed)
        model=RNN(matrix,len(classes)).to(device)
        opt=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=1e-3)
        lossf=nn.CrossEntropyLoss()
        Xt=torch.from_numpy(Xtr).long(); yt=torch.from_numpy(ytr)
        best,bstate,pat=-1,None,0
        for epoch in range(15):
            model.train()
            perm=torch.randperm(len(Xt),generator=torch.Generator().manual_seed(seed*313+epoch))
            for b in range(0,len(perm),128):
                sel=perm[b:b+128]; opt.zero_grad()
                lossf(model(Xt[sel].to(device)),yt[sel].to(device)).backward(); opt.step()
            f1=mf1(yva,predict(model,Xva))
            if f1>best+1e-4: best,pat=f1,0; bstate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:
                pat+=1
                if pat>=2: break
        model.load_state_dict(bstate)
        pr=predict(model,X[te])
        preds[(vk,seed)]=pr
        np.savez(os.path.join(HERE,f'nhtsaS_preds_{vk}_bilstm_s{seed}.npz'),pred=pr,y=y[te])
        res.append({"key":f"nhtsaS_{vk}_bilstm_s{seed}","f1":round(float(mf1(y[te],pr)),4)})
        log(res[-1])

# paired randomisation between fields, per run, Holm over the family of 9
rngp=np.random.default_rng(SEED)
def paired(yy,pa,pb,n=5000):
    d0=mf1(yy,pa)-mf1(yy,pb); cnt=0
    for _ in range(n):
        sw=rngp.random(len(yy))<0.5
        if abs(mf1(yy,np.where(sw,pb,pa))-mf1(yy,np.where(sw,pa,pb)))>=abs(d0)-1e-12: cnt+=1
    return d0,(cnt+1)/(n+1)
yy=y[te]; cons=[]
for s in (0,1,2):
    for a,b in [('summary','conseq'),('summary','remedy'),('remedy','conseq')]:
        d,p=paired(yy,preds[(a,s)],preds[(b,s)])
        cons.append({"contrast":f"shared bilstm s{s}: {a} vs {b}","delta":round(float(d),4),"p":round(float(p),4)})
        log(cons[-1])
cons_sorted=sorted(cons,key=lambda c:c["p"])
m=len(cons_sorted)
for i,c in enumerate(cons_sorted):
    c["p_holm"]=round(min(1.0,max((m-j)*cons_sorted[j]["p"] for j in range(i+1))),4)
json.dump({"results":res,"contrasts":cons},open(os.path.join(HERE,'control_nhtsa_shared.json'),'w'),indent=1)
print("NHTSA_SHARED DONE",flush=True)
