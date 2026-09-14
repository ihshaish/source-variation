"""Enumerates NHTSA recall campaigns through the public API by trying campaign
numbers of the form YYVnnn000 in order. Appends one JSON line per campaign
found to nhtsa_campaigns.jsonl and skips numbers already in the file, so a
stopped run can be restarted. Run: python3 nhtsa_crawl.py start_year end_year
(two-digit years)."""
import json
import os
import sys
import time
import urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'nhtsa_campaigns.jsonl')
seen = set()
if os.path.exists(OUT):
    seen = {json.loads(line)['NHTSACampaignNumber'] for line in open(OUT)}
year_start, year_end = int(sys.argv[1]), int(sys.argv[2])
found = miss = 0
with open(OUT, 'a') as out:
    for year in range(year_start, year_end + 1):
        streak = 0
        for number in range(1, 1000):
            campaign_number = f"{year:02d}V{number:03d}000"
            if campaign_number in seen:
                continue
            url = f"https://api.nhtsa.gov/recalls/campaignNumber?campaignNumber={campaign_number}"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                response = json.load(urllib.request.urlopen(req, timeout=20))
            except Exception:
                time.sleep(2)
                continue
            if response.get('Count', 0) > 0:
                result = response['results'][0]
                out.write(json.dumps({k: result.get(k) for k in
                    ('NHTSACampaignNumber', 'Component', 'Summary', 'Consequence', 'Remedy', 'ReportReceivedDate')}) + '\n')
                out.flush()
                found += 1
                streak = 0
            else:
                miss += 1
                streak += 1
                # forty consecutive misses means the year's numbering has run out
                if streak >= 40:
                    break
            time.sleep(0.15)
        print(f"year {year:02d}: cumulative found {found} miss {miss}", flush=True)
print("DONE", found)
