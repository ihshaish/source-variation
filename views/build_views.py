"""Attach synopsis and second-narrative views to the existing task records.

Reads the paper's task file and split untouched, joins the extra text fields
from the raw export by ACN, writes views_task.jsonl.gz with narr / syn / r2
per record. The task and split are exactly the paper's; only the text views
are new."""
import csv, glob, gzip, json, re
csv.field_size_limit(10**7)

BASE='/Users/Hisham/github_page/PhD_peter'
views={}
for f in sorted(glob.glob(f'{BASE}/data_asrs_v2/*.csv')):
    with open(f,errors='replace') as fh:
        r=csv.reader(fh); h1=next(r); h2=next(r)
        cols=[f"{a}/{b}".strip('/') for a,b in zip(h1,h2)]
        try:
            iacn=[i for i,c in enumerate(cols) if c.endswith('ACN')][0]
            i2=cols.index('Report 2/Narrative'); isyn=cols.index('Report 1/Synopsis')
        except (ValueError,IndexError):
            continue
        for row in r:
            if len(row)<=max(iacn,i2,isyn): continue
            acn=row[iacn].strip()
            if acn: views[acn]=(row[isyn].strip(), row[i2].strip())
print("view records:",len(views))

test=set(json.load(open(f'{BASE}/paper_a_nasa/l1_data_v2/split.json'))['test_acns'])
n=hit_s=hit_r2=0
with gzip.open(f'{BASE}/paper_a_nasa/l1_data_v2/task_aircraft.jsonl.gz','rt') as fin, \
     gzip.open(f'{BASE}/views_wip/views_task.jsonl.gz','wt') as fout:
    for l in fin:
        r=json.loads(l); n+=1
        syn,r2=views.get(r['acn'],('',''))
        if syn: hit_s+=1
        if len(r2)>40: hit_r2+=1
        fout.write(json.dumps({'acn':r['acn'],'label':r['label'],'year':r['year'],
                               'split':'test' if r['acn'] in test else 'train',
                               'narr':r['text'],'syn':syn,'r2':r2})+'\n')
print(f"task {n}, synopsis coverage {hit_s*100//n}%, dual-narrative {hit_r2} ({hit_r2*100//n}%)")
