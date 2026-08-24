"""Leg B end to end, from crawled campaigns: task build (top-level component
classes with 300+ support, exact-duplicate summaries confined to one side,
80/20 split by campaign), per-view word2vec, TF-IDF and BiLSTM per view,
three seeds, paired stats between views. Registered design."""
import glob,json,os,re,sys,time
import numpy as np
MASK_MODE=True
HERE=os.path.dirname(os.path.abspath(__file__))
TOKEN_RE=re.compile(r"[a-z][a-z0-9/-]+"); SEED=20260802
def log(*a): print(*a,flush=True)

rows=[json.loads(l) for l in open(os.path.join(HERE,'nhtsa_campaigns.jsonl'))]
seen=set(); camps=[]
for r in rows:
    if r['NHTSACampaignNumber'] in seen: continue
    seen.add(r['NHTSACampaignNumber']); camps.append(r)
log("campaigns:",len(camps))
from collections import Counter
def top(c): return (c or '').split(':')[0].split(',')[0].strip()
cnt=Counter(top(r['Component']) for r in camps)
classes=sorted([k for k,v in cnt.items() if v>=300 and k])
log("classes kept:",len(classes),classes)
lab={k:i for i,k in enumerate(classes)}
MASK=set()
for c in classes: MASK.update(re.findall(r'[a-z]+', c.lower()))
print("mask tokens:",sorted(MASK))
def strip_mask(s):
    return ' '.join(w for w in re.split(r'(\W+)', s or '') if w.lower() not in MASK)
data=[r for r in camps if top(r['Component']) in lab]
# exact-duplicate summaries to one side via grouping key
rng=np.random.default_rng(SEED)
# union-find over duplicate groups defined by ANY of the three fields
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
print("duplicate-union groups:",len(groups))
for r in data: r['split']='test' if r['NHTSACampaignNumber'] in test else 'train'
log("task:",len(data),"test",len(test))
y=np.array([lab[top(r['Component'])] for r in data]); te=np.array([r['split']=='test' for r in data])
json.dump({"classes":classes,"n":len(data),"n_test":int(te.sum())},open(os.path.join(HERE,'nhtsa3m_task_meta.json'),'w'))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
def mf1(yy,p): return f1_score(yy,p,average='macro')
VIEWS={'summary':'Summary','conseq':'Consequence','remedy':'Remedy'}
res=[]
# TF-IDF per view
for vk,field in VIEWS.items():
    v=TfidfVectorizer(lowercase=True,ngram_range=(1,2),min_df=3,sublinear_tf=True)
    Xtr=v.fit_transform([strip_mask(r[field]) for r,m in zip(data,~te) if m])
    Xte=v.transform([strip_mask(r[field]) for r,m in zip(data,te) if m])
    clf=LogisticRegression(max_iter=2000).fit(Xtr,y[~te])
    pr=clf.predict(Xte); prob=clf.predict_proba(Xte)
    np.savez(os.path.join(HERE,f'nhtsa3m_preds_{vk}_tfidf_s0.npz'),pred=pr,probs=prob,y=y[te])
    res.append({"key":f"nhtsa_{vk}_tfidf_s0","f1":round(float(mf1(y[te],pr)),4)})
    log(res[-1])
# w2v per view + BiLSTM
from gensim.models import Word2Vec
import torch, torch.nn as nn
device='mps' if torch.backends.mps.is_available() else 'cpu'
CAP=96; PAD,OOV=0,1
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
for vk,field in VIEWS.items():
    toks=[[w for w in TOKEN_RE.findall((r[field] or '').lower()) if w not in MASK][:CAP] for r in data]
    sents=[t for t,m in zip(toks,~te) if m]
    w2v=Word2Vec(vector_size=200,window=5,min_count=3,sg=1,epochs=10,workers=8,seed=SEED)
    w2v.build_vocab(sents); w2v.train(corpus_iterable=sents,total_examples=len(sents),epochs=10)
    vocab=set(t for ts in toks for t in ts)
    rng2=np.random.default_rng(0)
    idx,vecs={},[np.zeros(200,np.float32),rng2.normal(0,0.1,200).astype(np.float32)]
    for w in sorted(vocab):
        if w in w2v.wv: idx[w]=len(vecs); vecs.append(w2v.wv[w].astype(np.float32))
    matrix=np.stack(vecs)
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
        np.savez(os.path.join(HERE,f'nhtsa3m_preds_{vk}_bilstm_s{seed}.npz'),pred=pr,y=y[te])
        res.append({"key":f"nhtsa_{vk}_bilstm_s{seed}","f1":round(float(mf1(y[te],pr)),4)})
        log(res[-1])
json.dump(res,open(os.path.join(HERE,'nhtsa3m_results.json'),'w'),indent=1)
# paired randomisation between views (multiclass: swap predictions)
rngp=np.random.default_rng(SEED)
def paired(yy,pa,pb,n=5000):
    d0=mf1(yy,pa)-mf1(yy,pb); cnt=0
    for _ in range(n):
        sw=rngp.random(len(yy))<0.5
        if abs(mf1(yy,np.where(sw,pb,pa))-mf1(yy,np.where(sw,pa,pb)))>=abs(d0)-1e-12: cnt+=1
    return d0,(cnt+1)/(n+1)
cons=[]
pairs=[('summary','conseq'),('summary','remedy'),('remedy','conseq')]
for arch,seeds in (('tfidf',(0,)),('bilstm',(0,1,2))):
    for a,b in pairs:
        for s in seeds:
            da=np.load(os.path.join(HERE,f'nhtsa3m_preds_{a}_{arch}_s{s}.npz'))
            db=np.load(os.path.join(HERE,f'nhtsa3m_preds_{b}_{arch}_s{s}.npz'))
            d0,p=paired(da['y'],da['pred'],db['pred'])
            cons.append({"contrast":f"{arch} s{s}: {a} vs {b}","delta":round(float(d0),4),"p":round(float(p),4)})
            log(cons[-1])
m=len(cons)
for i,c in enumerate(sorted(cons,key=lambda c:c["p"])): c["p_holm"]=round(min(1.0,c["p"]*(m-i)),4)
json.dump(cons,open(os.path.join(HERE,'nhtsa3m_contrasts.json'),'w'),indent=1)
log("LEG B DONE")
