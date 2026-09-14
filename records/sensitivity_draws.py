# Feeds Supplementary Table S14 (five redrawn negative samples). Needs ASRS_DIR (the CSV export) to build the pool once; pool_records.jsonl.gz is not shipped (107 MB).
"""Pre-declared undersample sensitivity (register addendum 2026-08-26).
One negative-class draw per invocation: python3 sensitivity_draws.py <draw 1..5>.
Draw d: negatives resampled with seed 100+d from the same pool (Aircraft
positives fixed), same 80/20 stratified split policy, then word TF-IDF and
BiLSTM over draw-trained word2vec on narrative and synopsis, one training
each (frozen constants from build_task / records_word2vec / records_train).
Statistic: synopsis - narrative macro-F1 per model per draw. Appends to
sensitivity_draws.json. Build the pool cache first with argument 0."""
import csv,glob,gzip,json,os,re,sys,time
import numpy as np
csv.field_size_limit(10**7)
HERE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.dirname(HERE)
VIEWS=os.path.join(ROOT,'records'); NHTSA=os.path.join(ROOT,'nhtsa'); RESULTS=os.path.join(ROOT,'results')
POOL=os.path.join(HERE,'pool_records.jsonl.gz')
TOKEN_RE=re.compile(r"[a-z][a-z0-9/-]+")
MAX_YEAR=2021; PAD,OOV=0,1; CAPS={'narr':256,'syn':64}

def build_pool():
    syns={}
    for f in sorted(glob.glob(os.path.join(os.environ['ASRS_DIR'],'*.csv'))):
        with open(f,errors='replace') as fh:
            r=csv.reader(fh)
            try: h1=next(r); h2=next(r)
            except StopIteration: continue
            cols=[f"{a.strip()}/{b.strip()}" for a,b in zip(h1,h2)]
            try:
                iacn=[i for i,c in enumerate(cols) if c.endswith('ACN')][0]
                isyn=cols.index('Report 1/Synopsis')
            except (ValueError,IndexError): continue
            for row in r:
                if len(row)<=max(iacn,isyn): continue
                acn=row[iacn].strip()
                if acn: syns[acn]=row[isyn].strip()
    seen=set(); n=0
    with gzip.open(POOL,'wt') as out:
        for fn in sorted(f for f in os.listdir(os.environ['ASRS_DIR']) if f.endswith('.csv')):
            with open(os.path.join(os.environ['ASRS_DIR'],fn),newline='',encoding='utf-8',errors='replace') as fh:
                reader=csv.reader(fh)
                try: sections=next(reader); fields=next(reader)
                except StopIteration: continue
                header=[(s.strip(),t.strip()) for s,t in zip(sections,fields)]
                idx={}
                for key,sec,fld in [("date","Time","Date"),("primary","Assessments","Primary Problem"),
                                    ("narrative","Report 1","Narrative")]:
                    hits=[i for i,(s,t) in enumerate(header) if s==sec and t==fld]
                    if not hits and key=="primary":
                        hits=[i for i,(_,t) in enumerate(header) if t=="Primary Problem"]
                    idx[key]=hits[0] if hits else None
                if None in idx.values(): continue
                for row in reader:
                    if len(row)<=idx["narrative"] or not row[0].strip().isdigit(): continue
                    date=row[idx["date"]].strip()
                    if not re.fullmatch(r"(19|20)\d{4}",date): continue
                    year=int(date)//100
                    if year>MAX_YEAR: continue
                    acn=row[0].strip(); primary=row[idx["primary"]].strip(); narr=row[idx["narrative"]].strip()
                    if acn in seen or not primary or not narr: continue
                    seen.add(acn); n+=1
                    out.write(json.dumps({"acn":acn,"label":1 if primary=="Aircraft" else 0,
                                          "narr":narr,"syn":syns.get(acn,'')})+"\n")
    print("pool records:",n,flush=True)

def run_draw(d):
    import torch,torch.nn as nn
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
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
    print(f"draw {d}: task {len(data)} test {len(test)}",flush=True)
    y=np.array([r['label'] for r in data],np.int64)
    te=np.array([r['split']=='test' for r in data])
    res={"draw":d,"n":len(data),"n_test":int(te.sum())}
    # ---- TF-IDF ----
    for view in ('narr','syn'):
        texts=[r[view] for r in data]
        v=TfidfVectorizer(lowercase=True,ngram_range=(1,2),min_df=3,sublinear_tf=True)
        Xtr=v.fit_transform([t for t,m in zip(texts,~te) if m])
        Xte=v.transform([t for t,m in zip(texts,te) if m])
        clf=LogisticRegression(max_iter=2000,C=1.0).fit(Xtr,y[~te])
        p=(clf.predict_proba(Xte)[:,1]>=0.5).astype(int)
        res[f'tfidf_{view}']=round(float(f1_score(y[te],p,average='macro')),4)
        print(d,'tfidf',view,res[f'tfidf_{view}'],flush=True)
    # ---- per-view w2v + BiLSTM ----
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
        torch.manual_seed(100)
        model=RNN(matrix).to(device)
        opt=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=1e-3)
        lossf=nn.CrossEntropyLoss()
        Xtr_t=torch.from_numpy(Xtr).long(); ytr_t=torch.from_numpy(ytr)
        best,bstate,pat=-1.0,None,0
        for epoch in range(15):
            model.train()
            perm=torch.randperm(len(Xtr_t),generator=torch.Generator().manual_seed(100000+epoch))
            for b in range(0,len(perm),128):
                sel=perm[b:b+128]; opt.zero_grad()
                loss=lossf(model(Xtr_t[sel].to(device)),ytr_t[sel].to(device))
                loss.backward(); opt.step()
            f1=f1_score(yva,(predict(model,Xva)>=0.5).astype(int),average='macro')
            print(f"  d{d} {view} epoch {epoch} val {f1:.4f}",flush=True)
            if f1>best+1e-4: best,pat=f1,0; bstate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:
                pat+=1
                if pat>=2: break
        if bstate: model.load_state_dict(bstate)
        p=(predict(model,Xte_)>=0.5).astype(int)
        res[f'bilstm_{view}']=round(float(f1_score(yte_,p,average='macro')),4)
        print(d,'bilstm',view,res[f'bilstm_{view}'],flush=True)
        del model,matrix,X,Xtr_t
    res['tfidf_delta']=round(res['tfidf_syn']-res['tfidf_narr'],4)
    res['bilstm_delta']=round(res['bilstm_syn']-res['bilstm_narr'],4)
    out=[]
    rp=os.path.join(HERE,'sensitivity_draws.json')
    if os.path.exists(rp): out=json.load(open(rp))
    out.append(res); json.dump(out,open(rp,'w'),indent=1)
    print(json.dumps(res),flush=True)

if __name__=='__main__':
    d=int(sys.argv[1])
    if d==0: build_pool()
    else: run_draw(d)
