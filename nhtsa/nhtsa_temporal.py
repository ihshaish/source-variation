# Feeds Supplementary Table S17: train on campaigns filed 2000-2021, test on 2022-2026. Registered before it ran.
"""NHTSA temporal robustness (register 2026-08-27, pre-declared): same 16-class
recipe, split by campaign year — train 2000-2021, test 2022-2026; duplicate
groups spanning the boundary lose their test-side members. Per field: TF-IDF and
BiLSTM over field-trained word2vec (frozen constants), three trainings.
Outputs nhtsa_temporal.json. One invocation runs everything."""
import json,os,re
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.dirname(HERE)
VIEWS=os.path.join(ROOT,'views'); NHTSA=os.path.join(ROOT,'nhtsa'); RESULTS=os.path.join(ROOT,'results')
TOKEN_RE=re.compile(r"[a-z][a-z0-9/-]+")
PAD,OOV=0,1; CAP=96

def top(c): return (c or '').split(':')[0].split(',')[0].strip()

def build():
    seen=set(); camps=[]
    for l in open(f'{HERE}/nhtsa_campaigns.jsonl'):
        r=json.loads(l)
        if r['NHTSACampaignNumber'] in seen: continue
        seen.add(r['NHTSACampaignNumber']); camps.append(r)
    from collections import Counter
    cnt=Counter(top(r['Component']) for r in camps)
    classes=sorted([k for k,v in cnt.items() if v>=300 and k]); lab={k:i for i,k in enumerate(classes)}
    data=[r for r in camps if top(r['Component']) in lab]
    for r in data:
        r['year']=2000+int(r['NHTSACampaignNumber'][:2])
        r['split']='test' if r['year']>=2022 else 'train'
    # any-field exact-duplicate union-find; boundary-spanning groups lose test members
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
    for i,r in enumerate(data): groups.setdefault(find(i),[]).append(i)
    drop=set()
    for g in groups.values():
        sides={data[i]['split'] for i in g}
        if len(sides)>1:
            drop.update(i for i in g if data[i]['split']=='test')
    data=[r for i,r in enumerate(data) if i not in drop]
    y=np.array([lab[top(r['Component'])] for r in data]); te=np.array([r['split']=='test' for r in data])
    print(f"temporal task {len(data)} (dropped {len(drop)} boundary-dup test campaigns), test {int(te.sum())}",flush=True)
    return data,y,te,len(classes)

def main():
    import torch,torch.nn as nn
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from gensim.models import Word2Vec
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    data,y,te,C=build()
    out={"n":len(data),"n_test":int(te.sum())}
    class RNN(nn.Module):
        def __init__(s,m,C):
            super().__init__()
            s.emb=nn.Embedding.from_pretrained(torch.from_numpy(m),freeze=True,padding_idx=PAD)
            s.rnn=nn.LSTM(m.shape[1],64,batch_first=True,bidirectional=True)
            s.drop=nn.Dropout(0.3); s.fc=nn.Linear(128,C)
        def forward(s,x):
            h=s.rnn(s.emb(x))[1][0]
            return s.fc(s.drop(torch.cat([h[0],h[1]],dim=1)))
    def predict(model,X):
        model.eval(); outs=[]
        with torch.no_grad():
            for b in range(0,len(X),256):
                outs.append(model(torch.from_numpy(X[b:b+256]).long().to(device)).argmax(1).cpu().numpy())
        return np.concatenate(outs)
    for vk,field in (('summary','Summary'),('conseq','Consequence'),('remedy','Remedy')):
        texts=[(r[field] or '') for r in data]
        v=TfidfVectorizer(lowercase=True,ngram_range=(1,2),min_df=3,sublinear_tf=True)
        Xtr=v.fit_transform([t for t,m in zip(texts,~te) if m])
        Xte=v.transform([t for t,m in zip(texts,te) if m])
        clf=LogisticRegression(max_iter=2000,C=1.0).fit(Xtr,y[~te])
        out[f'tfidf_{vk}']=round(float(f1_score(y[te],clf.predict(Xte),average='macro')),4)
        print('tfidf',vk,out[f'tfidf_{vk}'],flush=True)
        sents=[TOKEN_RE.findall(t.lower()) for t,m in zip(texts,~te) if m and t.strip()]
        w2v=Word2Vec(vector_size=200,window=5,min_count=3,sg=1,epochs=10,workers=8,seed=20260802)
        w2v.build_vocab(sents); w2v.train(corpus_iterable=sents,total_examples=len(sents),epochs=10)
        tok_lists=[TOKEN_RE.findall(t.lower())[:CAP] for t in texts]
        vocab=set(t for ts in tok_lists for t in ts)
        r0=np.random.default_rng(0)
        idx,vecs={},[np.zeros(200,np.float32),r0.normal(0,0.1,200).astype(np.float32)]
        for w in sorted(vocab):
            if w in w2v.wv: idx[w]=len(vecs); vecs.append(w2v.wv[w].astype(np.float32))
        matrix=np.stack(vecs); del w2v
        X=np.zeros((len(tok_lists),CAP),np.int32)
        for i,ts in enumerate(tok_lists):
            for j,t in enumerate(ts): X[i,j]=idx.get(t,OOV)
        Xtr_all,ytr_all=X[~te],y[~te]; Xte_,yte_=X[te],y[te]
        for s in (0,1,2):
            Xtr2,Xva,ytr2,yva=train_test_split(Xtr_all,ytr_all,test_size=0.05,stratify=ytr_all,random_state=20260802+s)
            torch.manual_seed(100+s)
            model=RNN(matrix,C).to(device)
            opt=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=1e-3)
            lossf=nn.CrossEntropyLoss()
            Xtr_t=torch.from_numpy(Xtr2).long(); ytr_t=torch.from_numpy(ytr2)
            best,bstate,pat=-1.0,None,0
            for epoch in range(15):
                model.train()
                perm=torch.randperm(len(Xtr_t),generator=torch.Generator().manual_seed((100+s)*1000+epoch))
                for b in range(0,len(perm),128):
                    sel=perm[b:b+128]; opt.zero_grad()
                    loss=lossf(model(Xtr_t[sel].to(device)),ytr_t[sel].to(device))
                    loss.backward(); opt.step()
                f1=f1_score(yva,predict(model,Xva),average='macro')
                if f1>best+1e-4: best,pat=f1,0; bstate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                else:
                    pat+=1
                    if pat>=2: break
            if bstate: model.load_state_dict(bstate)
            out[f'bilstm_{vk}_s{s}']=round(float(f1_score(yte_,predict(model,Xte_),average='macro')),4)
            print('bilstm',vk,s,out[f'bilstm_{vk}_s{s}'],flush=True)
            del model
        del matrix,X,Xtr_t
        json.dump(out,open(f'{RESULTS}/nhtsa/nhtsa_temporal.json','w'),indent=1)
    json.dump(out,open(f'{RESULTS}/nhtsa/nhtsa_temporal.json','w'),indent=1)
    print("TEMPORAL DONE",flush=True)

if __name__=='__main__':
    main()
