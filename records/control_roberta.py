# Feeds Supplementary Tables S19 and S20 and Table 2's RoBERTa row: roberta-base fine-tuned per ASRS record and per NHTSA field, three seeds each. Registered before the ASRS runs finished (CLAIMS_REGISTER_v5.1_controls.md).
"""Contemporary-encoder control: roberta-base fine-tuned per record view.
ASRS views (narr cap 256 / syn cap 64) x 3 seeds and NHTSA fields (cap 96)
x 3 seeds under one fixed setup mirroring the reimplementation's
DistilBERT constants: AdamW 2e-5, batch 16, at most 3 epochs, early stop
(patience 1) on a 5% validation split, best weights, one held-out scoring.
Question, pre-committed: does the matched-view finding survive a modern
contextual classifier? Persists -> the view/field contrasts extend to
fine-tuned contemporary encoders; shrinks -> reported and scoped. GE is
excluded: the secure environment does not admit pretrained checkpoints."""
import gzip,json,os,re,sys
import numpy as np, torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
HERE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.dirname(HERE)
VIEWS=os.path.join(ROOT,'records'); NHTSA=os.path.join(ROOT,'nhtsa'); RESULTS=os.path.join(ROOT,'results')
TOKEN_RE=re.compile(r"[a-z][a-z0-9/-]+"); SEED=20260802
device='mps' if torch.backends.mps.is_available() else 'cpu'
tok=AutoTokenizer.from_pretrained('roberta-base')
def log(*a): print(*a,flush=True)

def run(texts,y,te,cap,nc,key,seed):
    Xtr_t,Xva_t,ytr,yva=train_test_split([t for t,m in zip(texts,~te) if m],y[~te],
        test_size=0.05,stratify=y[~te],random_state=SEED+seed)
    torch.manual_seed(900+seed)
    model=AutoModelForSequenceClassification.from_pretrained('roberta-base',num_labels=nc).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-5)
    def enc(batch): return tok(batch,truncation=True,max_length=cap,padding=True,return_tensors='pt')
    def predict(txts):
        model.eval(); outs=[]
        with torch.no_grad():
            for b in range(0,len(txts),64):
                x=enc(txts[b:b+64]).to(device)
                outs.append(torch.softmax(model(**x).logits,1).cpu().numpy())
        return np.concatenate(outs)
    def mf1(yy,p): return f1_score(yy,p.argmax(1),average='macro')
    best,bstate,pat=-1,None,0
    ytr_t=torch.from_numpy(np.asarray(ytr))
    for epoch in range(3):
        model.train()
        perm=torch.randperm(len(Xtr_t),generator=torch.Generator().manual_seed(seed*77+epoch))
        for b in range(0,len(perm),16):
            sel=perm[b:b+16].tolist()
            x=enc([Xtr_t[i] for i in sel]).to(device)
            loss=torch.nn.functional.cross_entropy(model(**x).logits,ytr_t[sel].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        f1=mf1(yva,predict(Xva_t))
        log(f"  {key} s{seed} epoch {epoch} val {f1:.4f}")
        if f1>best+1e-4:
            best,pat=f1,0
            bstate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            pat+=1
            if pat>=1: break
    model.load_state_dict(bstate)
    probs=predict([t for t,m in zip(texts,te) if m])
    f1=mf1(y[te],probs)
    np.savez(os.path.join(HERE,f'roberta_preds_{key}_s{seed}.npz'),
             probs=probs,pred=probs.argmax(1),y=y[te])
    del model; torch.mps.empty_cache() if device=='mps' else None
    return round(float(f1),4)

res=[]
# ---- NHTSA (cheap) ----------------------------------------------------------
rows=[json.loads(l) for l in open(os.path.join(NHTSA,'nhtsa_campaigns.jsonl'))]
seen=set(); camps=[]
for r in rows:
    if r['NHTSACampaignNumber'] in seen: continue
    seen.add(r['NHTSACampaignNumber']); camps.append(r)
from collections import Counter
def top(c): return (c or '').split(':')[0].split(',')[0].strip()
cnt=Counter(top(r['Component']) for r in camps)
classes=sorted([k for k,v in cnt.items() if v>=300 and k]); lab={k:i for i,k in enumerate(classes)}
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
yN=np.array([lab[top(r['Component'])] for r in data]); teN=np.array([r['split']=='test' for r in data])
log("NHTSA task:",len(data),"test",int(teN.sum()))
for vk,field in (('summary','Summary'),('conseq','Consequence'),('remedy','Remedy')):
    texts=[(r[field] or '') for r in data]
    for seed in (0,1,2):
        f1=run(texts,yN,teN,96,len(classes),f'nhtsa_{vk}',seed)
        res.append({"key":f"roberta_nhtsa_{vk}_s{seed}","f1":f1}); log(res[-1])
        json.dump(res,open(os.path.join(HERE,'control_roberta_partial.json'),'w'),indent=1)
# ---- ASRS views -------------------------------------------------------------
recs=[json.loads(l) for l in gzip.open(os.path.join(HERE,'records_task.jsonl.gz'),'rt')]
yA=np.array([int(r['label']) for r in recs]); teA=np.array([r['split']=='test' for r in recs])
log("ASRS task:",len(recs),"test",int(teA.sum()))
for view,cap in (('syn',64),('narr',256)):
    texts=[r[view] for r in recs]
    for seed in (0,1,2):
        f1=run(texts,yA,teA,cap,2,f'asrs_{view}',seed)
        res.append({"key":f"roberta_asrs_{view}_s{seed}","f1":f1}); log(res[-1])
        json.dump(res,open(os.path.join(HERE,'control_roberta_partial.json'),'w'),indent=1)

# ---- paired contrasts -------------------------------------------------------
rngp=np.random.default_rng(SEED)
def mf1c(yy,p): return f1_score(yy,p,average='macro')
def paired(yy,pa,pb,n=5000):
    d0=mf1c(yy,pa)-mf1c(yy,pb); cnt=0
    for _ in range(n):
        sw=rngp.random(len(yy))<0.5
        if abs(mf1c(yy,np.where(sw,pb,pa))-mf1c(yy,np.where(sw,pa,pb)))>=abs(d0)-1e-12: cnt+=1
    return round(float(d0),4),round((cnt+1)/(n+1),4)
cons=[]
for s in (0,1,2):
    a=np.load(os.path.join(HERE,f'roberta_preds_asrs_syn_s{s}.npz'))
    b=np.load(os.path.join(HERE,f'roberta_preds_asrs_narr_s{s}.npz'))
    d,p=paired(a['y'],a['pred'],b['pred'])
    cons.append({"contrast":f"roberta s{s}: syn vs narr","delta":d,"p":p}); log(cons[-1])
for s in (0,1,2):
    pr={vk:np.load(os.path.join(HERE,f'roberta_preds_nhtsa_{vk}_s{s}.npz')) for vk in ('summary','conseq','remedy')}
    for x,z in (('summary','conseq'),('summary','remedy'),('remedy','conseq')):
        d,p=paired(pr[x]['y'],pr[x]['pred'],pr[z]['pred'])
        cons.append({"contrast":f"roberta nhtsa s{s}: {x} vs {z}","delta":d,"p":p}); log(cons[-1])
json.dump({"results":res,"contrasts":cons},open(os.path.join(HERE,'control_roberta.json'),'w'),indent=1)
print("ROBERTA DONE",flush=True)
