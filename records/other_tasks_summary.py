"""Summarises other_tasks.jsonl per model: how many tasks favour the synopsis,
which intervals exclude zero, the median and range of the synopsis minus
narrative difference, and the per-task values. Reads other_tasks.jsonl and
writes other_tasks_summary.json next to this script.
Run: python3 other_tasks_summary.py"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

rows = [json.loads(line) for line in open(os.path.join(HERE, 'other_tasks.jsonl'))]
out = {}
for model in ('tfidf', 'bilstm'):
    scored = [r for r in rows if r['model'] == model and 'delta_mean' in r]
    per_task = {}
    for r in scored:
        entry = {'n_heldout': r['n_heldout'], 'narr': round(float(np.mean(r['narr_f1'])), 4),
                 'syn': round(float(np.mean(r['syn_f1'])), 4), 'delta': r['delta_mean'], 'ci95': r['ci95']}
        if 'delta_per_training' in r:
            entry['per_training'] = r['delta_per_training']
            entry['all_trainings_positive'] = all(v > 0 for v in r['delta_per_training'])
        per_task[r['category']] = entry
    summary = {'tasks_with_result': len(scored),
               'tasks_skipped': [r['category'] for r in rows if r['model'] == model and 'delta_mean' not in r],
               'positive_mean': sum(r['delta_mean'] > 0 for r in scored),
               'ci_above_zero': [r['category'] for r in scored if r['ci95'][0] > 0],
               'ci_below_zero': [r['category'] for r in scored if r['ci95'][1] < 0],
               'median_delta': None, 'range_delta': None, 'heldout_size_range': None,
               'per_task': per_task}
    if scored:
        deltas = [r['delta_mean'] for r in scored]
        sizes = [r['n_heldout'] for r in scored]
        summary['median_delta'] = round(float(np.median(deltas)), 4)
        summary['range_delta'] = [min(deltas), max(deltas)]
        summary['heldout_size_range'] = [min(sizes), max(sizes)]
    out[model] = summary
    if model == 'bilstm' and scored:
        large = [r for r in scored if r['n_heldout'] >= 1000]
        out[model]['heldout_ge_1000'] = {
            'tasks': len(large),
            'positive_mean': sum(r['delta_mean'] > 0 for r in large),
            'ci_above_zero': sum(r['ci95'][0] > 0 for r in large),
            'all_three_trainings_positive': sum(all(v > 0 for v in r['delta_per_training']) for r in large),
            'range_delta': [min(r['delta_mean'] for r in large), max(r['delta_mean'] for r in large)]}

json.dump(out, open(os.path.join(HERE, 'other_tasks_summary.json'), 'w'), indent=1)
for model, summary in out.items():
    print(model, json.dumps({k: v for k, v in summary.items() if k != 'per_task'}))
