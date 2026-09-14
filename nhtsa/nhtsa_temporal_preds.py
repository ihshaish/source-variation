# Feeds Supplementary Table S17: the temporal rerun with predictions kept, so the six declared Holm-corrected contrasts can be computed.
"""NHTSA temporal rerun with per-record predictions saved, so the pre-declared
Holm-corrected contrasts (register 2026-08-27 clarification b: six-contrast
family = {summary-consequence, summary-remedy} x 3 trainings) can be computed.
Identical setup and seeds to nhtsa_temporal.py; adds prediction persistence
and the paired approximate-randomisation tests in the views_stats.py convention
(p=(cnt+1)/(n+1), n=10000, Holm within the declared family). TF-IDF contrasts
are reported descriptively outside the family, per the declaration."""
import json,os
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.dirname(HERE)
VIEWS=os.path.join(ROOT,'views'); NHTSA=os.path.join(ROOT,'nhtsa'); RESULTS=os.path.join(ROOT,'results')
import sys; sys.path.insert(0,HERE)
from nhtsa_temporal import build,TOKEN_RE,PAD,OOV,CAP

def mf1(y,pred):
    from sklearn.metrics import f1_score
    return f1_score(y,pred,average='macro')

def paired(y,pa,pb,rng,n=10000):
    d0=mf1(y,pa)-mf1(y,pb); cnt=0
    for _ in range(n):
        sw=rng.random(len(y))<0.5
        qa=np.where(sw,pb,pa); qb=np.where(sw,pa,pb)
        if abs(mf1(y,qa)-mf1(y,qb))>=abs(d0)-1e-12: cnt+=1
    return d0,(cnt+1)/(n+1)

def main():
    import torch,torch.nn as nn
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from gensim.models import Word2Vec
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    data,y,te,C=build()
    preds={"y":y[te]}
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
        preds[f'tfidf_{vk}']=clf.predict(Xte)
        print('tfidf',vk,round(mf1(y[te],preds[f'tfidf_{vk}']),4),flush=True)
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
        from sklearn.metrics import f1_score
        import torch.nn as nn
        lossf=nn.CrossEntropyLoss()
        for s in (0,1,2):
            Xtr2,Xva,ytr2,yva=train_test_split(Xtr_all,ytr_all,test_size=0.05,stratify=ytr_all,random_state=20260802+s)
            torch.manual_seed(100+s)
            model=RNN(matrix,C).to(device)
            opt=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=1e-3)
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
            preds[f'bilstm_{vk}_s{s}']=predict(model,Xte_)
            print('bilstm',vk,s,round(mf1(yte_,preds[f'bilstm_{vk}_s{s}']),4),flush=True)
            del model
        del matrix,X,Xtr_t
    np.savez(os.path.join(HERE,'nhtsa_temporal_preds.npz'),**preds)
    # declared family: 2 contrasts x 3 trainings, Holm within six
    rng=np.random.default_rng(20260802)
    fam=[]
    for s in (0,1,2):
        for a,b in (('summary','conseq'),('summary','remedy')):
            d0,p=paired(preds['y'],preds[f'bilstm_{a}_s{s}'],preds[f'bilstm_{b}_s{s}'],rng)
            fam.append({"contrast":f"bilstm s{s}: {a}-{b}","delta":round(float(d0),4),"p":p})
    order=np.argsort([f['p'] for f in fam]); m=len(fam)
    running=0.0
    for rank,i in enumerate(order):
        adj=min(1.0,(m-rank)*fam[i]['p']); running=max(running,adj)
        fam[i]['p_holm']=round(running,4); fam[i]['p']=round(fam[i]['p'],5)
    extra=[]
    for a,b in (('summary','conseq'),('summary','remedy')):
        d0,p=paired(preds['y'],preds[f'tfidf_{a}'],preds[f'tfidf_{b}'],rng)
        extra.append({"contrast":f"tfidf: {a}-{b}","delta":round(float(d0),4),"p":round(p,5)})
    scores={k:round(float(mf1(preds['y'],preds[k])),4) for k in preds if k!='y'}
    json.dump({"n_test":int(len(preds['y'])),"scores":scores,"family_holm":fam,"tfidf_descriptive":extra},
              open(os.path.join(RESULTS,'nhtsa','nhtsa_temporal_tests.json'),'w'),indent=1)
    print("TEMPORAL TESTS DONE",flush=True)

if __name__=='__main__':
    main()
