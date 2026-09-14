"""Attaches the synopsis and the second narrative to the task records.
Reads the task file and split written by build_task.py, joins the two
extra text fields from the raw ASRS export by ACN, and writes
records_task.jsonl.gz with narr, syn and r2 per record. The task and the
split are unchanged; only the text fields are added.
Run: ASRS_DIR=<csv export> python3 build_records.py"""
import csv
import glob
import gzip
import json
import os

csv.field_size_limit(10**7)

HERE = os.path.dirname(os.path.abspath(__file__))
# ASRS_DIR is the CSV export; TASK_DIR is where build_task.py wrote the task
ASRS_DIR = os.environ['ASRS_DIR']
TASK_DIR = os.environ.get('TASK_DIR', os.path.join(os.path.dirname(HERE), 'asrs_pipeline', 'data'))

text_by_acn = {}
for path in sorted(glob.glob(os.path.join(ASRS_DIR, '*.csv'))):
    with open(path, errors='replace') as f:
        reader = csv.reader(f)
        header1 = next(reader)
        header2 = next(reader)
        columns = [f"{a}/{b}".strip('/') for a, b in zip(header1, header2)]
        try:
            acn_col = [i for i, c in enumerate(columns) if c.endswith('ACN')][0]
            r2_col = columns.index('Report 2/Narrative')
            syn_col = columns.index('Report 1/Synopsis')
        except (ValueError, IndexError):
            continue
        for row in reader:
            if len(row) <= max(acn_col, r2_col, syn_col):
                continue
            acn = row[acn_col].strip()
            if acn:
                text_by_acn[acn] = (row[syn_col].strip(), row[r2_col].strip())
print("export records:", len(text_by_acn))

test_acns = set(json.load(open(os.path.join(TASK_DIR, 'split.json')))['test_acns'])
count = with_synopsis = with_second = 0
with gzip.open(os.path.join(TASK_DIR, 'task_aircraft.jsonl.gz'), 'rt') as fin, \
     gzip.open(os.path.join(HERE, 'records_task.jsonl.gz'), 'wt') as fout:
    for line in fin:
        record = json.loads(line)
        count += 1
        syn, r2 = text_by_acn.get(record['acn'], ('', ''))
        if syn:
            with_synopsis += 1
        if len(r2) > 40:
            with_second += 1
        fout.write(json.dumps({'acn': record['acn'], 'label': record['label'], 'year': record['year'],
                               'split': 'test' if record['acn'] in test_acns else 'train',
                               'narr': record['text'], 'syn': syn, 'r2': r2}) + '\n')
print(f"task {count}, synopsis coverage {with_synopsis*100//count}%, dual-narrative {with_second} ({with_second*100//count}%)")
