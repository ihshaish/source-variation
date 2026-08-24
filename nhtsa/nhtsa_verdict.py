"""Go/no-go quantities from crawled campaigns, per the registered criteria:
>=8 top-level component classes with enough support (scaled to the pilot
window), median field lengths >=15 tokens."""
import json,re,statistics,sys
from collections import Counter
TOKEN_RE=re.compile(r"[a-z][a-z0-9/-]+")
rows=[json.loads(l) for l in open('/Users/Hisham/github_page/PhD_peter/views_wip/nhtsa_campaigns.jsonl')]
print("campaigns:",len(rows))
top=Counter((r['Component'] or '').split(':')[0].split(',')[0].strip() for r in rows)
print("top-level component classes:",len(top))
for k,v in top.most_common(14): print(f"  {v:4d}  {k}")
for f in ('Summary','Consequence','Remedy'):
    L=[len(TOKEN_RE.findall((r[f] or '').lower())) for r in rows]
    print(f"{f}: median {statistics.median(L):.0f} tokens, empty {sum(1 for x in L if x==0)}")
