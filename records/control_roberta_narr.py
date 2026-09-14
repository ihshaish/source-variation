# Feeds Supplementary Table S19: one ASRS narrative fine-tune per invocation (one process per run keeps a 16 GB machine out of swap).
"""One ASRS narrative RoBERTa fine-tune per process invocation (memory
hygiene: twelve in-process runs exhausted a 16GB machine into swap).
Identical setup to control_roberta.py: AdamW 2e-5, batch 16, <=3 epochs,
early stop patience 1 on a 5% validation split, cap 256, one held-out
scoring. Usage: python3 control_roberta_narr.py <seed>"""
import gzip,json,os,sys
import numpy as np, torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
HERE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.dirname(HERE)
VIEWS=os.path.join(ROOT,'records'); NHTSA=os.path.join(ROOT,'nhtsa'); RESULTS=os.path.join(ROOT,'results')
SEED=20260802; seed=int(sys.argv[1])
device='mps' if torch.backends.mps.is_available() else 'cpu'
tok=AutoTokenizer.from_pretrained('roberta-base')
recs=[json.loads(l) for l in gzip.open(os.path.join(HERE,'records_task.jsonl.gz'),'rt')]
y=np.array([int(r['label']) for r in recs]); te=np.array([r['split']=='test' for r in recs])
texts=[r['narr'] for r in recs]; cap=256
print(f"narr s{seed}: task {len(recs)} test {int(te.sum())}",flush=True)
Xtr_t,Xva_t,ytr,yva=train_test_split([t for t,m in zip(texts,~te) if m],y[~te],
    test_size=0.05,stratify=y[~te],random_state=SEED+seed)
torch.manual_seed(900+seed)
model=AutoModelForSequenceClassification.from_pretrained('roberta-base',num_labels=2).to(device)
opt=torch.optim.AdamW(model.parameters(),lr=2e-5)
def enc(b): return tok(b,truncation=True,max_length=cap,padding=True,return_tensors='pt')
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
    print(f"  asrs_narr s{seed} epoch {epoch} val {f1:.4f}",flush=True)
    if f1>best+1e-4:
        best,pat=f1,0
        bstate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    else:
        pat+=1
        if pat>=1: break
model.load_state_dict(bstate)
probs=predict([t for t,m in zip(texts,te) if m])
f1=mf1(y[te],probs)
np.savez(os.path.join(HERE,f'roberta_preds_asrs_narr_s{seed}.npz'),
         probs=probs,pred=probs.argmax(1),y=y[te])
res=json.load(open(os.path.join(HERE,'control_roberta_partial.json')))
res.append({"key":f"roberta_asrs_narr_s{seed}","f1":round(float(f1),4)})
json.dump(res,open(os.path.join(HERE,'control_roberta_partial.json'),'w'),indent=1)
print(json.dumps(res[-1]),flush=True)
