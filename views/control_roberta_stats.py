# Feeds Supplementary Table S19: paired contrasts over the stored RoBERTa predictions, run once all fifteen fine-tunes exist.
"""Final paired contrasts for the RoBERTa suite, from stored predictions.
Runs after all 15 fine-tunes. Emits control_roberta.json + ROBERTA DONE."""
import json,os
import numpy as np
from sklearn.metrics import f1_score
HERE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.dirname(HERE)
VIEWS=os.path.join(ROOT,'views'); NHTSA=os.path.join(ROOT,'nhtsa'); RESULTS=os.path.join(ROOT,'results')
rngp=np.random.default_rng(20260802)
def mf1c(yy,p): return f1_score(yy,p,average='macro')
def paired(yy,pa,pb,n=5000):
    d0=mf1c(yy,pa)-mf1c(yy,pb); cnt=0
    for _ in range(n):
        sw=rngp.random(len(yy))<0.5
        if abs(mf1c(yy,np.where(sw,pb,pa))-mf1c(yy,np.where(sw,pa,pb)))>=abs(d0)-1e-12: cnt+=1
    return round(float(d0),4),round((cnt+1)/(n+1),4)
res=json.load(open(os.path.join(HERE,'control_roberta_partial.json')))
cons=[]
for s in (0,1,2):
    a=np.load(os.path.join(HERE,f'roberta_preds_asrs_syn_s{s}.npz'))
    b=np.load(os.path.join(HERE,f'roberta_preds_asrs_narr_s{s}.npz'))
    d,p=paired(a['y'],a['pred'],b['pred'])
    cons.append({"contrast":f"roberta s{s}: syn vs narr","delta":d,"p":p}); print(cons[-1],flush=True)
for s in (0,1,2):
    pr={vk:np.load(os.path.join(HERE,f'roberta_preds_nhtsa_{vk}_s{s}.npz')) for vk in ('summary','conseq','remedy')}
    for x,z in (('summary','conseq'),('summary','remedy'),('remedy','conseq')):
        d,p=paired(pr[x]['y'],pr[x]['pred'],pr[z]['pred'])
        cons.append({"contrast":f"roberta nhtsa s{s}: {x} vs {z}","delta":d,"p":p}); print(cons[-1],flush=True)
json.dump({"results":res,"contrasts":cons},open(os.path.join(RESULTS,'views','control_roberta.json'),'w'),indent=1)
print("ROBERTA DONE",flush=True)
