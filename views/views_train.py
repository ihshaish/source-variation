"""Final models for one view x one embedding, three seeds, on the paper's
exact task and split. Narrative models are additionally scored on the
dual-report subset reading reporter 2's account, paired by record.
python3 views_train.py --view narr|syn --emb glove200|w2vview [--smoke]"""
import argparse, gzip, json, os, re, time
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors

HERE=os.path.dirname(os.path.abspath(__file__))
TOKEN_RE=re.compile(r"[a-z][a-z0-9/-]+")
SEED=20260802; PAD,OOV=0,1
CAPS={'narr':256,'syn':64,'r2':256}
GLOVE='/Users/Hisham/github_page/PhD_peter/embeddings/glove.6B.200d.txt'

def toks(t,cap): return TOKEN_RE.findall(t.lower())[:cap]

def load(view):
    recs=[]
    with gzip.open(os.path.join(HERE,'views_task.jsonl.gz'),'rt') as f:
        for l in f:
            r=json.loads(l)
            r['toks']=toks(r[view],CAPS[view])
            r['r2toks']=toks(r['r2'],CAPS['r2']) if view=='narr' and len(r['r2'])>40 else None
            recs.append(r)
    return recs

def build_matrix(emb,view,vocab):
    rng=np.random.default_rng(0)
    if emb=='glove200':
        idx,vecs={},[np.zeros(200,np.float32),rng.normal(0,0.1,200).astype(np.float32)]
        for line in open(GLOVE,encoding='utf-8'):
            p=line.rstrip().split(' ')
            if p[0] in vocab:
                idx[p[0]]=len(vecs); vecs.append(np.asarray(p[1:],np.float32))
        return idx,np.stack(vecs)
    kv=KeyedVectors.load(os.path.join(HERE,f"w2v_{view}_200d.kv"),mmap='r')
    idx,vecs={},[np.zeros(200,np.float32),rng.normal(0,0.1,200).astype(np.float32)]
    for w in sorted(vocab):
        if w in kv: idx[w]=len(vecs); vecs.append(kv[w].astype(np.float32))
    return idx,np.stack(vecs)

def encode(tok_lists,idx,cap):
    X=np.zeros((len(tok_lists),cap),np.int32)
    for i,ts in enumerate(tok_lists):
        for j,t in enumerate(ts): X[i,j]=idx.get(t,OOV)
    return X

class RNN(nn.Module):
    def __init__(s,m):
        super().__init__()
        s.emb=nn.Embedding.from_pretrained(torch.from_numpy(m),freeze=True,padding_idx=PAD)
        s.rnn=nn.LSTM(m.shape[1],64,batch_first=True,bidirectional=True)
        s.drop=nn.Dropout(0.3); s.fc=nn.Linear(128,2)
    def forward(s,x):
        h=s.rnn(s.emb(x))[1][0]
        return s.fc(s.drop(torch.cat([h[0],h[1]],dim=1)))

def predict(model,X,device):
    model.eval(); outs=[]
    with torch.no_grad():
        for b in range(0,len(X),256):
            outs.append(torch.softmax(model(torch.from_numpy(X[b:b+256]).long().to(device)),1)[:,1].cpu().numpy())
    return np.concatenate(outs)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--view',required=True,choices=['narr','syn'])
    ap.add_argument('--emb',required=True,choices=['glove200','w2vview'])
    ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args()
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    recs=load(a.view)
    if a.smoke:
        rng=np.random.default_rng(0); recs=[recs[i] for i in rng.choice(len(recs),3000,replace=False)]
    vocab=set(t for r in recs for t in r['toks'])
    if a.view=='narr': vocab|=set(t for r in recs if r['r2toks'] for t in r['r2toks'])
    idx,matrix=build_matrix(a.emb,a.view,vocab)
    cap=CAPS[a.view]
    X=encode([r['toks'] for r in recs],idx,cap)
    y=np.array([int(r['label']) for r in recs],np.int64)
    is_test=np.array([r['split']=='test' for r in recs])
    acns=np.array([r['acn'] for r in recs])
    Xtr_all,ytr_all=X[~is_test],y[~is_test]; Xte,yte=X[is_test],y[is_test]
    dual=None
    if a.view=='narr':
        di=[i for i in range(len(recs)) if is_test[i] and recs[i]['r2toks']]
        Xr2=encode([recs[i]['r2toks'] for i in di],idx,CAPS['r2'])
        Xr1=X[di]; yd=y[di]; dacn=acns[di]; dual=(Xr1,Xr2,yd,dacn)
        print(f"dual-report held-out records: {len(di)}")
    tag='smoke_' if a.smoke else ''
    done=set()
    rp=os.path.join(HERE,'views_results.jsonl')
    if os.path.exists(rp):
        done={json.loads(l)['key'] for l in open(rp)}
    for seed in ((0,) if a.smoke else (0,1,2)):
        key=f"{tag}final_{a.view}_{a.emb}_s{seed}"
        if key in done: print("skip",key); continue
        Xtr,Xva,ytr,yva=train_test_split(Xtr_all,ytr_all,test_size=0.05,stratify=ytr_all,random_state=SEED+seed)
        torch.manual_seed(100+seed)
        model=RNN(matrix).to(device)
        opt=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=1e-3)
        lossf=nn.CrossEntropyLoss()
        Xtr_t=torch.from_numpy(Xtr).long(); ytr_t=torch.from_numpy(ytr)
        best,bstate,pat=-1.0,None,0; t0=time.time()
        for epoch in range(1 if a.smoke else 15):
            model.train()
            perm=torch.randperm(len(Xtr_t),generator=torch.Generator().manual_seed((100+seed)*1000+epoch))
            for b in range(0,len(perm),128):
                sel=perm[b:b+128]; opt.zero_grad()
                loss=lossf(model(Xtr_t[sel].to(device)),ytr_t[sel].to(device))
                loss.backward(); opt.step()
            f1=f1_score(yva,(predict(model,Xva,device)>=0.5).astype(int),average='macro')
            if f1>best+1e-4: best,pat=f1,0; bstate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:
                pat+=1
                if pat>=2: break
        if bstate: model.load_state_dict(bstate)
        pte=predict(model,Xte,device)
        f1=f1_score(yte,(pte>=0.5).astype(int),average='macro')
        np.savez(os.path.join(HERE,f"{tag}preds_{a.view}_{a.emb}_s{seed}.npz"),probs=pte,y=yte,acns=acns[is_test])
        out={"key":key,"view":a.view,"emb":a.emb,"seed":seed,"test_macro_f1":round(float(f1),4),
             "val_f1":round(float(best),4),"secs":round(time.time()-t0)}
        if dual is not None:
            Xr1,Xr2,yd,dacn=dual
            p1=predict(model,Xr1,device); p2=predict(model,Xr2,device)
            np.savez(os.path.join(HERE,f"{tag}dualpreds_{a.emb}_s{seed}.npz"),p1=p1,p2=p2,y=yd,acns=dacn)
            out["dual_r1_f1"]=round(float(f1_score(yd,(p1>=0.5).astype(int),average='macro')),4)
            out["dual_r2_f1"]=round(float(f1_score(yd,(p2>=0.5).astype(int),average='macro')),4)
        with open(rp,'a') as f: f.write(json.dumps(out)+'\n')
        print(json.dumps(out))

if __name__=='__main__': main()
