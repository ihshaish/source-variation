"""Seed-averaged primary contrasts with joint record-resampling
confidence intervals, the matched-length threshold sensitivity on the
dual-report subset, and the seed-averaged interaction D. Arithmetic on
stored predictions only; nothing trains. Reads the preds_*.npz,
dualpreds_*.npz and roberta_preds_*.npz files here and the
nhtsa_preds_*.npz files in nhtsa/, and writes results/records/seedavg.json.
Run: python3 seedavg_and_threshold.py"""
import gzip
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


def macro_f1(y, pred):
    return f1_score(y, pred, average='macro')


def pred(data):
    if 'pred' in data:
        return np.asarray(data['pred'])
    probs = data['probs']
    return probs.argmax(1) if probs.ndim == 2 else (probs >= 0.5).astype(int)


def ci_of(fun, y, n, seed):
    rng = np.random.default_rng(20260802 + seed)
    boot = []
    for _ in range(B):
        index = rng.integers(0, n, n)
        boot.append(fun(index))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


out = {}
syn_preds = [pred(np.load(f'{HERE}/preds_syn_w2vview_s{s}.npz')) for s in (0, 1, 2)]
narr_preds = [pred(np.load(f'{HERE}/preds_narr_w2vview_s{s}.npz')) for s in (0, 1, 2)]
y = np.load(f'{HERE}/preds_syn_w2vview_s0.npz')['y']
n = len(y)
mean_delta = np.mean([macro_f1(y, syn_preds[i]) - macro_f1(y, narr_preds[i]) for i in range(3)])
ci = ci_of(lambda ix: np.mean([macro_f1(y[ix], syn_preds[i][ix]) - macro_f1(y[ix], narr_preds[i][ix]) for i in range(3)]), y, n, 1)
out['syn_narr_bilstm'] = {'mean': round(float(mean_delta), 4), 'ci': ci}
print('syn-narr BiLSTM w2v', out['syn_narr_bilstm'])
syn_preds = [pred(np.load(f'{HERE}/roberta_preds_asrs_syn_s{s}.npz')) for s in (0, 1, 2)]
narr_preds = [pred(np.load(f'{HERE}/roberta_preds_asrs_narr_s{s}.npz')) for s in (0, 1, 2)]
mean_delta = np.mean([macro_f1(y, syn_preds[i]) - macro_f1(y, narr_preds[i]) for i in range(3)])
ci = ci_of(lambda ix: np.mean([macro_f1(y[ix], syn_preds[i][ix]) - macro_f1(y[ix], narr_preds[i][ix]) for i in range(3)]), y, n, 2)
out['syn_narr_roberta'] = {'mean': round(float(mean_delta), 4), 'ci': ci}
print('syn-narr RoBERTa', out['syn_narr_roberta'])
records = [json.loads(line) for line in gzip.open(f'{HERE}/records_task.jsonl.gz', 'rt')]
by_acn = {r['acn']: r for r in records if r.get('r2')}
dual0 = np.load(f'{HERE}/dualpreds_w2vview_s0.npz')
acns = dual0['acns']
yd = dual0['y']
nd = len(yd)
ratio = np.array([len(by_acn[a]['r2'].split()) / max(1, len(by_acn[a]['narr'].split())) for a in acns])
preds_r1 = [(np.load(f'{HERE}/dualpreds_w2vview_s{s}.npz')['p1'] >= 0.5).astype(int) for s in (0, 1, 2)]
preds_r2 = [(np.load(f'{HERE}/dualpreds_w2vview_s{s}.npz')['p2'] >= 0.5).astype(int) for s in (0, 1, 2)]
mean_delta = np.mean([macro_f1(yd, preds_r1[i]) - macro_f1(yd, preds_r2[i]) for i in range(3)])
ci = ci_of(lambda ix: np.mean([macro_f1(yd[ix], preds_r1[i][ix]) - macro_f1(yd[ix], preds_r2[i][ix]) for i in range(3)]), yd, nd, 3)
out['dual_raw'] = {'mean': round(float(mean_delta), 4), 'ci': ci}
print('dual raw w2v', out['dual_raw'])
matched = ratio >= 0.7
y_matched = yd[matched]
n_matched = int(matched.sum())
mean_delta = np.mean([macro_f1(y_matched, preds_r1[i][matched]) - macro_f1(y_matched, preds_r2[i][matched]) for i in range(3)])
ci = ci_of(lambda ix: np.mean([macro_f1(y_matched[ix], preds_r1[i][matched][ix]) - macro_f1(y_matched[ix], preds_r2[i][matched][ix]) for i in range(3)]), y_matched, n_matched, 4)
out['dual_matched'] = {'mean': round(float(mean_delta), 4), 'ci': ci, 'n': n_matched}
print('dual matched', out['dual_matched'])
# thresholds over both embeddings and three seeds, six runs
r1_runs = {}
r2_runs = {}
for emb in ('glove200', 'w2vview'):
    for s in (0, 1, 2):
        data = np.load(f'{HERE}/dualpreds_{emb}_s{s}.npz')
        r1_runs[(emb, s)] = (data['p1'] >= 0.5).astype(int)
        r2_runs[(emb, s)] = (data['p2'] >= 0.5).astype(int)
out['thresholds'] = []
for cutoff in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
    mask = ratio >= cutoff
    y_cut = yd[mask]
    runs = [round(float(macro_f1(y_cut, r1_runs[k][mask]) - macro_f1(y_cut, r2_runs[k][mask])), 4) for k in r1_runs]
    out['thresholds'].append({'cutoff': cutoff, 'n': int(mask.sum()), 'min': min(runs), 'max': max(runs)})
    print(out['thresholds'][-1])
pool_syn = [pred(np.load(f'{HERE}/preds_syn_meanmlp_s{s}.npz')) for s in (0, 1, 2)]
pool_narr = [pred(np.load(f'{HERE}/preds_narr_meanmlp_s{s}.npz')) for s in (0, 1, 2)]
seq_syn = [pred(np.load(f'{HERE}/preds_syn_w2vview_s{s}.npz')) for s in (0, 1, 2)]
seq_narr = [pred(np.load(f'{HERE}/preds_narr_w2vview_s{s}.npz')) for s in (0, 1, 2)]
D_per_seed = [round(float((macro_f1(y, seq_syn[i]) - macro_f1(y, pool_syn[i])) - (macro_f1(y, seq_narr[i]) - macro_f1(y, pool_narr[i]))), 4) for i in range(3)]
print('D per seed:', D_per_seed)
mean_delta = np.mean(D_per_seed)
ci = ci_of(lambda ix: np.mean([(macro_f1(y[ix], seq_syn[i][ix]) - macro_f1(y[ix], pool_syn[i][ix])) - (macro_f1(y[ix], seq_narr[i][ix]) - macro_f1(y[ix], pool_narr[i][ix])) for i in range(3)]), y, n, 5)
out['D_viewtrained'] = {'mean': round(float(mean_delta), 4), 'ci': ci, 'per_seed': D_per_seed}
print('D avg', out['D_viewtrained'])
y_nhtsa = np.load(f'{NHTSA}/nhtsa_preds_summary_bilstm_s0.npz')['y']
n_nhtsa = len(y_nhtsa)
field_preds = {field: [pred(np.load(f'{NHTSA}/nhtsa_preds_{field}_bilstm_s{s}.npz')) for s in (0, 1, 2)] for field in ('summary', 'conseq', 'remedy')}
for other, tag, seed in (('conseq', 'sum_conseq', 6), ('remedy', 'sum_remedy', 7)):
    mean_delta = np.mean([macro_f1(y_nhtsa, field_preds['summary'][i]) - macro_f1(y_nhtsa, field_preds[other][i]) for i in range(3)])
    ci = ci_of(lambda ix: np.mean([macro_f1(y_nhtsa[ix], field_preds['summary'][i][ix]) - macro_f1(y_nhtsa[ix], field_preds[other][i][ix]) for i in range(3)]), y_nhtsa, n_nhtsa, seed)
    out[tag] = {'mean': round(float(mean_delta), 4), 'ci': ci}
    print(tag, out[tag])
json.dump(out, open(f'{RESULTS}/records/seedavg.json', 'w'), indent=1)
print('DONE')
