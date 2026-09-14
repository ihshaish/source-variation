# Feeds Supplementary Table S14: BiLSTM trainings 2 and 3 per redraw, after sensitivity_draws.py.
"""Negative-draw completion (register 2026-08-27): BiLSTM trainings 2 and 3
(torch seeds 101, 102) per draw, identical setup to sensitivity_draws.py.
python3 sensitivity_extra.py <draw 1..5> <torchseed>"""
import gzip,json,os,re,sys
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.dirname(HERE)
VIEWS=os.path.join(ROOT,'records'); NHTSA=os.path.join(ROOT,'nhtsa'); RESULTS=os.path.join(ROOT,'results')
POOL=os.path.join(HERE,'pool_records.jsonl.gz')
TOKEN_RE=re.compile(r"[a-z][a-z0-9/-]+")
PAD,OOV=0,1; CAPS={'narr':256,'syn':64}

def main(d,tseed):
    import torch,torch.nn as nn
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    from gensim.models import Word2Vec
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    recs=[json.loads(l) for l in gzip.open(POOL,'rt')]
    pos=[r for r in recs if r['label']==1]; neg=[r for r in recs if r['label']==0]
    rng=np.random.default_rng(100+d)
    keep=rng.choice(len(neg),size=len(pos),replace=False)
    data=pos+[neg[i] for i in sorted(keep)]
    order=rng.permutation(len(data)); data=[data[i] for i in order]
    test=set()
    for lbl in (0,1):
        cls=[r['acn'] for r in data if r['label']==lbl]
        co=rng.permutation(len(cls)); ntest=round(0.2*len(cls))
        test.update(cls[i] for i in co[:ntest])
    for r in data: r['split']='test' if r['acn'] in test else 'train'
    y=np.array([r['label'] for r in data],np.int64)
    te=np.array([r['split']=='test' for r in data])
    res={"draw":d,"torch_seed":tseed}
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
    for view in ('narr','syn'):
        cap=CAPS[view]
        sents=[TOKEN_RE.findall(r[view].lower()) for r in data if r['split']=='train' and r[view].strip()]
        w2v=Word2Vec(vector_size=200,window=5,min_count=3,sg=1,epochs=10,workers=8,seed=20260802)
        w2v.build_vocab(sents); w2v.train(corpus_iterable=sents,total_examples=len(sents),epochs=10)
        tok_lists=[TOKEN_RE.findall(r[view].lower())[:cap] for r in data]
        vocab=set(t for ts in tok_lists for t in ts)
        r0=np.random.default_rng(0)
        idx,vecs={},[np.zeros(200,np.float32),r0.normal(0,0.1,200).astype(np.float32)]
        for w in sorted(vocab):
            if w in w2v.wv: idx[w]=len(vecs); vecs.append(w2v.wv[w].astype(np.float32))
        matrix=np.stack(vecs); del w2v
        X=np.zeros((len(tok_lists),cap),np.int32)
        for i,ts in enumerate(tok_lists):
            for j,t in enumerate(ts): X[i,j]=idx.get(t,OOV)
        Xtr_all,ytr_all=X[~te],y[~te]; Xte_,yte_=X[te],y[te]
        Xtr,Xva,ytr,yva=train_test_split(Xtr_all,ytr_all,test_size=0.05,stratify=ytr_all,random_state=20260802)
        torch.manual_seed(tseed)
        model=RNN(matrix).to(device)
        opt=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=1e-3)
        lossf=nn.CrossEntropyLoss()
        Xtr_t=torch.from_numpy(Xtr).long(); ytr_t=torch.from_numpy(ytr)
        best,bstate,pat=-1.0,None,0
        for epoch in range(15):
            model.train()
            perm=torch.randperm(len(Xtr_t),generator=torch.Generator().manual_seed(tseed*1000+epoch))
            for b in range(0,len(perm),128):
                sel=perm[b:b+128]; opt.zero_grad()
                loss=lossf(model(Xtr_t[sel].to(device)),ytr_t[sel].to(device))
                loss.backward(); opt.step()
            from sklearn.metrics import f1_score as f1s
            f1=f1s(yva,(predict(model,Xva)>=0.5).astype(int),average='macro')
            print(f"  d{d} t{tseed} {view} epoch {epoch} val {f1:.4f}",flush=True)
            if f1>best+1e-4: best,pat=f1,0; bstate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:
                pat+=1
                if pat>=2: break
        if bstate: model.load_state_dict(bstate)
        from sklearn.metrics import f1_score as f1s
        p=(predict(model,Xte_)>=0.5).astype(int)
        res[f'bilstm_{view}']=round(float(f1s(yte_,p,average='macro')),4)
        del model,matrix,X,Xtr_t
    res['bilstm_delta']=round(res['bilstm_syn']-res['bilstm_narr'],4)
    rp=os.path.join(HERE,'sensitivity_extra.json')
    out=json.load(open(rp)) if os.path.exists(rp) else []
    out.append(res); json.dump(out,open(rp,'w'),indent=1)
    print(json.dumps(res),flush=True)

if __name__=='__main__':
    main(int(sys.argv[1]),int(sys.argv[2]))
