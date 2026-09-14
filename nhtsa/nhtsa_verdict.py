"""Prints the quantities that decide whether the crawled campaigns support
the task: the number of top-level component classes and the support of the
fourteen largest, and the median token length and empty count of each of
the summary, consequence and remedy fields. Reads nhtsa_campaigns.jsonl.
Run: python3 nhtsa_verdict.py"""
import json
import os
import re
import statistics
from collections import Counter
TOKEN_RE = re.compile(r"[a-z][a-z0-9/-]+")
records = [json.loads(line) for line in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nhtsa_campaigns.jsonl'))]
print("campaigns:", len(records))
top_counts = Counter((record['Component'] or '').split(':')[0].split(',')[0].strip() for record in records)
print("top-level component classes:", len(top_counts))
for name, count in top_counts.most_common(14):
    print(f"  {count:4d}  {name}")
for field in ('Summary', 'Consequence', 'Remedy'):
    lengths = [len(TOKEN_RE.findall((record[field] or '').lower())) for record in records]
    print(f"{field}: median {statistics.median(lengths):.0f} tokens, empty {sum(1 for x in lengths if x == 0)}")
