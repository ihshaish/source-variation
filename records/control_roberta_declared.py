"""Computes three RoBERTa contrasts from stored predictions: the
interaction D_R = [S(syn, RoBERTa) - S(syn, BiLSTM)] - [S(narr, RoBERTa) -
S(narr, BiLSTM)] with a record bootstrap per seed, once against the
BiLSTM and once against TF-IDF (seed 0); the NHTSA gains G_field =
S(field, RoBERTa) - S(field, BiLSTM) with a record bootstrap of G_remedy
minus G_summary per seed; and a paired permutation test of TF-IDF against
RoBERTa on the NHTSA summary field per seed. Reads the roberta_preds_*.npz
and preds_*.npz files here and the nhtsa_preds_*.npz files in nhtsa/, and
writes results/records/control_roberta_declared.json.
Run: python3 control_roberta_declared.py"""
import json
import os
import numpy as np
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RECORDS = os.path.join(ROOT, 'records')
NHTSA = os.path.join(ROOT, 'nhtsa')
RESULTS = os.path.join(ROOT, 'results')
B = 2000


def macro_f1(yy, pred):
    return f1_score(yy, pred, average='macro')


def pred(data):
    if 'pred' in data:
        return data['pred']
    probs = data['probs']
    return probs.argmax(1) if probs.ndim == 2 else (probs >= 0.5).astype(int)


def paired(yy, pred_a, pred_b, rng, n=5000):
    delta = macro_f1(yy, pred_a) - macro_f1(yy, pred_b)
    count = 0
    for _ in range(n):
        swap = rng.random(len(yy)) < 0.5
        if abs(macro_f1(yy, np.where(swap, pred_b, pred_a)) - macro_f1(yy, np.where(swap, pred_a, pred_b))) >= abs(delta) - 1e-12:
            count += 1
    return round(float(delta), 4), round((count + 1) / (n + 1), 4)


out = {}
out['DR'] = []
for seed in (0, 1, 2):
    syn_roberta = np.load(os.path.join(HERE, f'roberta_preds_asrs_syn_s{seed}.npz'))
    narr_roberta = np.load(os.path.join(HERE, f'roberta_preds_asrs_narr_s{seed}.npz'))
    y = syn_roberta['y']
    for base, tag in ((f'w2vview_s{seed}', 'bilstm'), ('tfidf_s0', 'tfidf')):
        syn_base = np.load(os.path.join(HERE, f'preds_syn_{base}.npz'))
        narr_base = np.load(os.path.join(HERE, f'preds_narr_{base}.npz'))
        pred_syn_r, pred_narr_r, pred_syn_b, pred_narr_b = pred(syn_roberta), pred(narr_roberta), pred(syn_base), pred(narr_base)
        delta = (macro_f1(y, pred_syn_r) - macro_f1(y, pred_syn_b)) - (macro_f1(y, pred_narr_r) - macro_f1(y, pred_narr_b))
        rng = np.random.default_rng(20260802 + seed)
        boot = []
        for _ in range(B):
            index = rng.integers(0, len(y), len(y))
            yy = y[index]
            boot.append((macro_f1(yy, pred_syn_r[index]) - macro_f1(yy, pred_syn_b[index])) - (macro_f1(yy, pred_narr_r[index]) - macro_f1(yy, pred_narr_b[index])))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out['DR'].append({"seed": seed, "baseline": tag, "D_R": round(float(delta), 4),
                          "ci": [round(float(lo), 4), round(float(hi), 4)]})
        print(out['DR'][-1], flush=True)
out['nhtsa_gains'] = []
for seed in (0, 1, 2):
    roberta = {field: pred(np.load(os.path.join(HERE, f'roberta_preds_nhtsa_{field}_s{seed}.npz'))) for field in ('summary', 'remedy')}
    bilstm = {field: pred(np.load(os.path.join(NHTSA, f'nhtsa_preds_{field}_bilstm_s{seed}.npz'))) for field in ('summary', 'remedy')}
    y = np.load(os.path.join(HERE, f'roberta_preds_nhtsa_summary_s{seed}.npz'))['y']
    delta = (macro_f1(y, roberta['remedy']) - macro_f1(y, bilstm['remedy'])) - (macro_f1(y, roberta['summary']) - macro_f1(y, bilstm['summary']))
    rng = np.random.default_rng(20260802 + seed)
    boot = []
    for _ in range(B):
        index = rng.integers(0, len(y), len(y))
        yy = y[index]
        boot.append((macro_f1(yy, roberta['remedy'][index]) - macro_f1(yy, bilstm['remedy'][index])) - (macro_f1(yy, roberta['summary'][index]) - macro_f1(yy, bilstm['summary'][index])))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    gain_summary = round(float(macro_f1(y, roberta['summary']) - macro_f1(y, bilstm['summary'])), 4)
    gain_remedy = round(float(macro_f1(y, roberta['remedy']) - macro_f1(y, bilstm['remedy'])), 4)
    out['nhtsa_gains'].append({"seed": seed, "G_summary": gain_summary, "G_remedy": gain_remedy,
        "G_remedy_minus_G_summary": round(float(delta), 4),
        "ci": [round(float(lo), 4), round(float(hi), 4)]})
    print(out['nhtsa_gains'][-1], flush=True)
out['nhtsa_summary_tfidf_vs_roberta'] = []
tfidf = np.load(os.path.join(NHTSA, 'nhtsa_preds_summary_tfidf_s0.npz'))
for seed in (0, 1, 2):
    roberta = np.load(os.path.join(HERE, f'roberta_preds_nhtsa_summary_s{seed}.npz'))
    rng = np.random.default_rng(20260802 + seed)
    delta, p = paired(roberta['y'], pred(tfidf), pred(roberta), rng)
    out['nhtsa_summary_tfidf_vs_roberta'].append(
        {"seed": seed, "delta_tfidf_minus_roberta": delta, "p": p})
    print(out['nhtsa_summary_tfidf_vs_roberta'][-1], flush=True)
json.dump(out, open(os.path.join(RESULTS, 'records', 'control_roberta_declared.json'), 'w'), indent=1)
print("DECLARED DONE", flush=True)
