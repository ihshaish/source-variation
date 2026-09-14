"""Paired contrasts for the RoBERTa runs, from stored predictions, once
all fifteen fine-tunes exist: synopsis against narrative per seed on
ASRS, and the three field pairs per seed on NHTSA. Reads
control_roberta_partial.json and the roberta_preds_*.npz files here and
writes results/records/control_roberta.json.
Run: python3 control_roberta_stats.py"""
import json
import os
import numpy as np
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RECORDS = os.path.join(ROOT, 'records')
NHTSA = os.path.join(ROOT, 'nhtsa')
RESULTS = os.path.join(ROOT, 'results')
rng = np.random.default_rng(20260802)


def macro_f1(yy, pred):
    return f1_score(yy, pred, average='macro')


def paired(yy, pred_a, pred_b, n=5000):
    delta = macro_f1(yy, pred_a) - macro_f1(yy, pred_b)
    count = 0
    for _ in range(n):
        swap = rng.random(len(yy)) < 0.5
        if abs(macro_f1(yy, np.where(swap, pred_b, pred_a)) - macro_f1(yy, np.where(swap, pred_a, pred_b))) >= abs(delta) - 1e-12:
            count += 1
    return round(float(delta), 4), round((count + 1) / (n + 1), 4)


results = json.load(open(os.path.join(HERE, 'control_roberta_partial.json')))
contrasts = []
for seed in (0, 1, 2):
    syn = np.load(os.path.join(HERE, f'roberta_preds_asrs_syn_s{seed}.npz'))
    narr = np.load(os.path.join(HERE, f'roberta_preds_asrs_narr_s{seed}.npz'))
    delta, p = paired(syn['y'], syn['pred'], narr['pred'])
    contrasts.append({"contrast": f"roberta s{seed}: syn vs narr", "delta": delta, "p": p})
    print(contrasts[-1], flush=True)
for seed in (0, 1, 2):
    preds = {field: np.load(os.path.join(HERE, f'roberta_preds_nhtsa_{field}_s{seed}.npz')) for field in ('summary', 'conseq', 'remedy')}
    for left, right in (('summary', 'conseq'), ('summary', 'remedy'), ('remedy', 'conseq')):
        delta, p = paired(preds[left]['y'], preds[left]['pred'], preds[right]['pred'])
        contrasts.append({"contrast": f"roberta nhtsa s{seed}: {left} vs {right}", "delta": delta, "p": p})
        print(contrasts[-1], flush=True)
json.dump({"results": results, "contrasts": contrasts}, open(os.path.join(RESULTS, 'records', 'control_roberta.json'), 'w'), indent=1)
print("ROBERTA DONE", flush=True)
