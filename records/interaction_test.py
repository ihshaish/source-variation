"""Computes the record-by-model interaction D = (sequence minus pool |
synopsis) minus (sequence minus pool | narrative) with a record-level
bootstrap interval over the shared held-out set, per seed, in the
word2vec family; then the paired TF-IDF against BiLSTM contrast per
record type; then the ensemble family with Holm correction. Reads the
preds_*.npz files in this folder and writes interaction_test.json.
Run: python3 interaction_test.py"""
import json
import os
import numpy as np
from sklearn.metrics import f1_score

rng = np.random.default_rng(20260802)


def macro_f1(y, probs):
    return f1_score(y, (probs >= 0.5).astype(int), average='macro')


def load(name):
    data = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), name))
    return data['y'], data['probs']


out = {}
print("== interaction D = (seq-pool|syn) - (seq-pool|narr), w2v family:", flush=True)
results = []
for seed in (0, 1, 2):
    y, seq_syn = load(f'preds_syn_w2vview_s{seed}.npz')
    _, pool_syn = load(f'preds_syn_meanmlp_s{seed}.npz')
    _, seq_narr = load(f'preds_narr_w2vview_s{seed}.npz')
    _, pool_narr = load(f'preds_narr_meanmlp_s{seed}.npz')
    D0 = (macro_f1(y, seq_syn) - macro_f1(y, pool_syn)) - (macro_f1(y, seq_narr) - macro_f1(y, pool_narr))
    n = len(y)
    index = np.arange(n)
    values = []
    for _ in range(5000):
        sample = rng.choice(index, n, replace=True)
        values.append((macro_f1(y[sample], seq_syn[sample]) - macro_f1(y[sample], pool_syn[sample])) - (macro_f1(y[sample], seq_narr[sample]) - macro_f1(y[sample], pool_narr[sample])))
    lo, hi = np.percentile(values, [2.5, 97.5])
    results.append({"seed": seed, "D": round(float(D0), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
                    "excludes_zero": bool(lo > 0 or hi < 0)})
    print(results[-1], flush=True)
out['interaction'] = results
print("== ranking-reversal check: TF-IDF vs BiLSTM(w2v) per record, paired:", flush=True)


def paired(y, probs_a, probs_b, n=5000):
    delta = macro_f1(y, probs_a) - macro_f1(y, probs_b)
    count = 0
    for _ in range(n):
        swap = rng.random(len(y)) < 0.5
        if abs(macro_f1(y, np.where(swap, probs_b, probs_a)) - macro_f1(y, np.where(swap, probs_a, probs_b))) >= abs(delta) - 1e-12:
            count += 1
    return delta, (count + 1) / (n + 1)


reversal = []
for record in ('narr', 'syn'):
    y_t, tfidf_probs = load(f'preds_{record}_tfidf_s0.npz')
    for seed in (0, 1, 2):
        _, bilstm_probs = load(f'preds_{record}_w2vview_s{seed}.npz')
        delta, p = paired(y_t, tfidf_probs, bilstm_probs)
        reversal.append({"view": record, "seed": seed, "tfidf_minus_bilstm": round(float(delta), 4), "p": round(float(p), 4)})
        print(reversal[-1], flush=True)
out['reversal'] = reversal
print("== ensemble family, Holm:", flush=True)
ensemble = []
for emb in ('glove200', 'w2vview'):
    for seed in (0, 1, 2):
        y, narr_probs = load(f'preds_narr_{emb}_s{seed}.npz')
        _, syn_probs = load(f'preds_syn_{emb}_s{seed}.npz')
        ensemble_probs = (narr_probs + syn_probs) / 2
        best = syn_probs if macro_f1(y, syn_probs) >= macro_f1(y, narr_probs) else narr_probs
        delta, p = paired(y, ensemble_probs, best)
        ensemble.append({"emb": emb, "seed": seed, "delta": round(float(delta), 4), "p": round(float(p), 4)})
m = len(ensemble)
for i, contrast in enumerate(sorted(ensemble, key=lambda c: c["p"])):
    contrast["p_holm"] = round(min(1.0, contrast["p"] * (m - i)), 4)
for contrast in ensemble:
    print(contrast, flush=True)
out['ensemble_holm'] = ensemble
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'interaction_test.json'), 'w'), indent=1)
print("INTERACTION DONE", flush=True)
