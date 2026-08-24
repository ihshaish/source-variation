"""Enumerate NHTSA recall campaigns by number pattern (YYVnnn000) via the
public API. Appends one JSON line per found campaign; resume-safe. Args:
start_year end_year (2-digit)."""
import json, os, sys, time, urllib.request
HERE=os.path.dirname(os.path.abspath(__file__))
OUT=os.path.join(HERE,'nhtsa_campaigns.jsonl')
seen=set()
if os.path.exists(OUT):
    seen={json.loads(l)['NHTSACampaignNumber'] for l in open(OUT)}
y0,y1=int(sys.argv[1]),int(sys.argv[2])
found=miss=0
with open(OUT,'a') as out:
    for yy in range(y0,y1+1):
        streak=0
        for n in range(1,1000):
            cn=f"{yy:02d}V{n:03d}000"
            if cn in seen: continue
            url=f"https://api.nhtsa.gov/recalls/campaignNumber?campaignNumber={cn}"
            try:
                req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0"})
                d=json.load(urllib.request.urlopen(req,timeout=20))
            except Exception:
                time.sleep(2); continue
            if d.get('Count',0)>0:
                r=d['results'][0]
                out.write(json.dumps({k:r.get(k) for k in
                    ('NHTSACampaignNumber','Component','Summary','Consequence','Remedy','ReportReceivedDate')})+'\n')
                out.flush(); found+=1; streak=0
            else:
                miss+=1; streak+=1
                if streak>=40: break   # past the last campaign of the year
            time.sleep(0.15)
        print(f"year {yy:02d}: cumulative found {found} miss {miss}", flush=True)
print("DONE", found)
