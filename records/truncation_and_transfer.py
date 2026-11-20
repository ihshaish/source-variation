"""Further paired contrasts on the stored ASRS and NHTSA predictions: how many
records exceed the length caps and the synopsis minus narrative contrast on
records within and beyond the cap, the narrative refit at a 512-token cap,
RoBERTa against the BiLSTM, the dual-report transfer by length ratio, the
shared-embedding contrasts, the interaction D and the redrawn tasks.
Arithmetic on stored predictions only; nothing trains. Every estimate is the
mean over three trainings of the within-training macro-F1 difference, with a
percentile interval from 2000 joint resamples of the held-out cases.
Reads the prediction files written by records/, asrs_pipeline/ and nhtsa/,
records_task.jsonl.gz and results/records/, and writes
results/records/truncation_and_transfer.json.
Run: python3 truncation_and_transfer.py"""
import gzip
import json
import os
import re

import numpy as np
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NHTSA = os.path.join(ROOT, 'nhtsa')
ASRS = os.path.join(ROOT, 'asrs_pipeline')
RESULTS = os.path.join(ROOT, 'results', 'records')
TOKEN_RE = re.compile(r"[a-z][a-z0-9/-]+")
B = 2000
SEED0 = 20260802


def r6(value):
    return round(float(value), 6)


def pred(path):
    data = np.load(path)
    if 'pred' in data.files:
        return np.asarray(data['pred'])
    probs = data['probs']
    if probs.ndim == 2:
        return probs.argmax(1)
    return (probs >= 0.5).astype(int)


def f1_weighted(weights, y, preds, classes):
    """Macro-F1 of each prediction vector under each row of case weights
    (resamples x cases). Equals f1_score(average='macro') on the resampled
    arrays."""
    truth = np.stack([(y == c) for c in classes], 1).astype(float)
    truth_counts = weights @ truth
    out = []
    for p in preds:
        predicted = np.stack([(p == c) for c in classes], 1).astype(float)
        predicted_counts = weights @ predicted
        true_pos = weights @ (predicted * truth)
        denominator = predicted_counts + truth_counts
        present = denominator > 0
        f1 = np.where(present, 2 * true_pos / np.where(present, denominator, 1), 0.0)
        out.append(f1.sum(1) / present.sum(1))
    return np.stack(out, 1)


def boot_f1(y, preds, seed, mask=None):
    """Observed macro-F1 per vector and B resampled macro-F1 values per vector,
    one shared index draw per resample."""
    if mask is not None:
        y = y[mask]
        preds = [p[mask] for p in preds]
    n = len(y)
    classes = np.unique(np.concatenate([y] + list(preds)))
    observed = np.array([f1_score(y, p, average='macro') for p in preds])
    rng = np.random.default_rng(SEED0 + seed)
    resampled = []
    for start in range(0, B, 200):
        draws = [np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(min(200, B - start))]
        resampled.append(f1_weighted(np.stack(draws).astype(float), y, preds, classes))
    return y, observed, np.concatenate(resampled)


def contrast(y, runs_a, runs_b, seed, mask=None):
    """Mean over trainings of macro-F1(A) minus macro-F1(B)."""
    n_runs = len(runs_a)
    y_used, observed, boot = boot_f1(y, list(runs_a) + list(runs_b), seed, mask)
    per_training = observed[:n_runs] - observed[n_runs:]
    distribution = (boot[:, :n_runs] - boot[:, n_runs:]).mean(1)
    lo, hi = np.percentile(distribution, [2.5, 97.5])
    n_pos = int((y_used == 1).sum()) if y_used.max() == 1 else None
    return {'n': int(len(y_used)), 'n_pos': n_pos,
            'mean': r6(per_training.mean()), 'ci95': [r6(lo), r6(hi)],
            'per_training': [r6(v) for v in per_training],
            'range': [r6(per_training.min()), r6(per_training.max())],
            'f1_A': [r6(v) for v in observed[:n_runs]], 'f1_B': [r6(v) for v in observed[n_runs:]],
            'B': B, 'rng_seed': SEED0 + seed}


def interaction(y, seq_1, pool_1, seq_2, pool_2, seed, mask=None):
    """D = mean over trainings of (sequence minus pooling on record 1) minus
    (sequence minus pooling on record 2)."""
    y_used, observed, boot = boot_f1(y, list(seq_1) + list(pool_1) + list(seq_2) + list(pool_2), seed, mask)
    observed = observed.reshape(4, 3)
    boot = boot.reshape(B, 4, 3)
    per_training = (observed[0] - observed[1]) - (observed[2] - observed[3])
    distribution = ((boot[:, 0] - boot[:, 1]) - (boot[:, 2] - boot[:, 3])).mean(1)
    lo, hi = np.percentile(distribution, [2.5, 97.5])
    return {'n': int(len(y_used)), 'mean': r6(per_training.mean()), 'ci95': [r6(lo), r6(hi)],
            'per_training': [r6(v) for v in per_training], 'B': B, 'rng_seed': SEED0 + seed}


def rate(mask):
    return {'n': int(mask.sum()), 'of': int(len(mask)), 'share': r6(mask.mean())}


def three(pattern):
    return [P[pattern.format(s=s)] for s in range(3)]


out = {'procedure': 'mean over three trainings of the within-training macro-F1 difference; percentile 95% interval '
                    'from 2000 joint resamples of held-out cases conditional on the trained models; '
                    'numpy default_rng(20260802 + k), k given per entry as rng_seed'}

records = [json.loads(line) for line in gzip.open(os.path.join(HERE, 'records_task.jsonl.gz'), 'rt')]
is_test = np.array([r['split'] == 'test' for r in records])
acn_test = np.array([r['acn'] for r in records])[is_test]
y = np.array([int(r['label']) for r in records])[is_test]
narr_len_all = np.array([len(TOKEN_RE.findall(r['narr'].lower())) for r in records])
syn_len_all = np.array([len(TOKEN_RE.findall(r['syn'].lower())) for r in records])
narr_len = narr_len_all[is_test]
syn_len = syn_len_all[is_test]

files = {}
for emb in ('w2vview', 'glove200', 'meanmlp', 'meanmlpglove'):
    for record in ('narr', 'syn'):
        for s in range(3):
            files[f'{record}_{emb}_s{s}'] = os.path.join(HERE, f'preds_{record}_{emb}_s{s}.npz')
for record in ('narr', 'syn'):
    files[f'{record}_tfidf'] = os.path.join(HERE, f'preds_{record}_tfidf_s0.npz')
    files[f'{record}_charngram'] = os.path.join(HERE, f'preds_{record}_charngram_s0.npz')
    for s in range(3):
        files[f'roberta_{record}_s{s}'] = os.path.join(HERE, f'roberta_preds_asrs_{record}_s{s}.npz')
for s in range(3):
    files[f'cap512_glove200_s{s}'] = os.path.join(ASRS, 'results_ablate', f'cap512_preds_glove200_bilstm_s{s}.npz')
    files[f'cap256_glove200_s{s}'] = os.path.join(ASRS, 'results', f'preds_glove200_bilstm_s{s}.npz')
alignment = {}
for key, path in files.items():
    data = np.load(path)
    acns_equal = bool(np.array_equal(data['acns'], acn_test)) if 'acns' in data.files else 'not stored'
    alignment[key] = {'n': int(len(data['y'])), 'labels_equal_task_order': bool(np.array_equal(data['y'], y)),
                      'acns_equal_task_order': acns_equal}
# the cap-512 refit and its cap-256 counterpart score the asrs_pipeline task file in its own order
task_acns = [json.loads(line)['acn'] for line in gzip.open(os.path.join(ASRS, 'data', 'task_aircraft.jsonl.gz'), 'rt')]
split_acns = set(json.load(open(os.path.join(ASRS, 'data', 'split.json')))['test_acns'])
alignment['asrs_pipeline_test_order_equals_records_task_test_order'] = [a for a in task_acns if a in split_acns] == list(acn_test)
out['alignment'] = alignment
P = {key: pred(path) for key, path in files.items()}

stored = json.load(open(os.path.join(RESULTS, 'seedavg.json')))
reproduced = {'syn_narr_bilstm': contrast(y, three('syn_w2vview_s{s}'), three('narr_w2vview_s{s}'), 1),
              'syn_narr_roberta': contrast(y, three('roberta_syn_s{s}'), three('roberta_narr_s{s}'), 2),
              'D_viewtrained': interaction(y, three('syn_w2vview_s{s}'), three('syn_meanmlp_s{s}'),
                                           three('narr_w2vview_s{s}'), three('narr_meanmlp_s{s}'), 5)}
out['reproduction_of_seedavg'] = {key: {'recomputed': {'mean': value['mean'], 'ci95': value['ci95']},
                                        'stored': stored[key]} for key, value in reproduced.items()}

truncation = {'rule': "tokens = re.findall('[a-z][a-z0-9/-]+', text.lower()); the model reads the first 256 "
                      "narrative tokens and the first 64 synopsis tokens; a record is truncated when it has more "
                      "tokens than its cap",
              'narrative_gt256': {'task': rate(narr_len_all > 256), 'train': rate(narr_len_all[~is_test] > 256),
                                  'heldout': rate(narr_len > 256)},
              'narrative_le256_heldout': rate(narr_len <= 256),
              'synopsis_gt64': {'task': rate(syn_len_all > 64), 'heldout': rate(syn_len > 64)},
              'narrative_gt512_heldout': rate(narr_len > 512),
              'narrative_tokens_heldout_pct': {q: int(np.percentile(narr_len, q)) for q in (50, 75, 90, 95)},
              'synopsis_tokens_heldout_pct': {q: int(np.percentile(syn_len, q)) for q in (50, 75, 90, 95, 99)},
              'share_of_heldout_narrative_tokens_beyond_cap': r6(np.clip(narr_len - 256, 0, None).sum() / narr_len.sum()),
              'truncated_share_by_class_heldout': {'aircraft': r6((narr_len > 256)[y == 1].mean()),
                                                   'other': r6((narr_len > 256)[y == 0].mean())}}
within = narr_len <= 256
within_both = within & (syn_len <= 64)
truncation['subset_narr_le256'] = {'n': int(within.sum()), 'aircraft': int(y[within].sum()),
                                   'other': int((1 - y[within]).sum())}
truncation['subset_narr_le256_and_syn_le64'] = {'n': int(within_both.sum()), 'aircraft': int(y[within_both].sum()),
                                                'other': int((1 - y[within_both]).sum())}
truncation['subset_truncated'] = {'n': int((~within).sum()), 'aircraft': int(y[~within].sum()),
                                  'other': int((1 - y[~within]).sum())}
within_roberta = None
try:
    os.environ['HF_HUB_OFFLINE'] = '1'
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('roberta-base', local_files_only=True)
    test_records = [r for r, test in zip(records, is_test) if test]
    narr_bpe = np.array([len(ids) for ids in tokenizer([r['narr'] for r in test_records], truncation=False,
                                                       add_special_tokens=True)['input_ids']])
    syn_bpe = np.array([len(ids) for ids in tokenizer([r['syn'] for r in test_records], truncation=False,
                                                      add_special_tokens=True)['input_ids']])
    within_roberta = (narr_bpe <= 256) & (syn_bpe <= 64)
    truncation['roberta_bpe_heldout'] = {
        'rule': 'roberta-base byte-pair tokens incl. <s> </s>, max_length 256 / 64 (control_roberta.py)',
        'narrative_gt256': rate(narr_bpe > 256), 'synopsis_gt64': rate(syn_bpe > 64),
        'subset_both_within_cap': {'n': int(within_roberta.sum()), 'aircraft': int(y[within_roberta].sum())}}
except Exception as error:
    truncation['roberta_bpe_heldout'] = f'not computed: {error}'
out['truncation'] = truncation

families = {'bilstm_w2v': (three('syn_w2vview_s{s}'), three('narr_w2vview_s{s}')),
            'bilstm_glove200': (three('syn_glove200_s{s}'), three('narr_glove200_s{s}')),
            'roberta': (three('roberta_syn_s{s}'), three('roberta_narr_s{s}')),
            'meanpool_w2v': (three('syn_meanmlp_s{s}'), three('narr_meanmlp_s{s}')),
            'meanpool_glove200': (three('syn_meanmlpglove_s{s}'), three('narr_meanmlpglove_s{s}')),
            'tfidf_word': ([P['syn_tfidf']], [P['narr_tfidf']]),
            'tfidf_char': ([P['syn_charngram']], [P['narr_charngram']])}
by_truncation = {}
for i, (name, (synopsis, narrative)) in enumerate(families.items()):
    by_truncation[name] = {'all_heldout': contrast(y, synopsis, narrative, 100 + i),
                           'narr_le256': contrast(y, synopsis, narrative, 120 + i, within),
                           'narr_le256_and_syn_le64': contrast(y, synopsis, narrative, 140 + i, within_both),
                           'narr_gt256': contrast(y, synopsis, narrative, 160 + i, ~within)}
if within_roberta is not None:
    by_truncation['roberta']['bpe_both_within_cap'] = contrast(y, *families['roberta'], 180, within_roberta)
six = by_truncation['bilstm_w2v']['narr_le256']['per_training'] + by_truncation['bilstm_glove200']['narr_le256']['per_training']
by_truncation['six_bilstm_runs_narr_le256_range'] = [min(six), max(six)]
out['synopsis_minus_narrative_by_truncation'] = by_truncation
out['interaction_D_by_truncation'] = {
    'all': reproduced['D_viewtrained'],
    'narr_le256': interaction(y, three('syn_w2vview_s{s}'), three('syn_meanmlp_s{s}'),
                              three('narr_w2vview_s{s}'), three('narr_meanmlp_s{s}'), 205, within)}
out['narrative_cap512_refit'] = {
    'narr_cap512_minus_narr_cap256': contrast(y, three('cap512_glove200_s{s}'), three('cap256_glove200_s{s}'), 303),
    'narr_cap512_minus_narr_cap256_on_narr_gt256': contrast(y, three('cap512_glove200_s{s}'),
                                                            three('cap256_glove200_s{s}'), 304, ~within),
    'syn_glove200_minus_narr_cap512': contrast(y, three('syn_glove200_s{s}'), three('cap512_glove200_s{s}'), 301),
    'syn_glove200_minus_narr_cap512_on_narr_gt256': contrast(y, three('syn_glove200_s{s}'),
                                                             three('cap512_glove200_s{s}'), 305, ~within),
    'note': 'the synopsis models come from records/records_train.py and the narrative models from '
            'asrs_pipeline/train.py and ablations.py; they share task, split, tokeniser and GloVe-200 '
            'but not every training detail'}

bilstm = by_truncation['bilstm_w2v']['all_heldout']
roberta = by_truncation['roberta']['all_heldout']
versus = {'stored_seedavg': {key: stored[key] for key in ('syn_narr_bilstm', 'syn_narr_roberta')},
          'recomputed': {'bilstm_w2v': out['reproduction_of_seedavg']['syn_narr_bilstm']['recomputed'],
                         'roberta': out['reproduction_of_seedavg']['syn_narr_roberta']['recomputed']},
          'bootstrap_streams': 'BiLSTM rng 20260802+1, RoBERTa rng 20260802+2 (different resample draws)',
          'per_training_f1': {'bilstm_syn': bilstm['f1_A'], 'bilstm_narr': bilstm['f1_B'],
                              'roberta_syn': roberta['f1_A'], 'roberta_narr': roberta['f1_B']},
          'per_training_delta': {'bilstm_w2v': bilstm['per_training'], 'roberta': roberta['per_training']},
          'heldout_prediction_disagreement_bilstm_vs_roberta': {
              'narrative': [r6((P[f'narr_w2vview_s{s}'] != P[f'roberta_narr_s{s}']).mean()) for s in range(3)],
              'synopsis': [r6((P[f'syn_w2vview_s{s}'] != P[f'roberta_syn_s{s}']).mean()) for s in range(3)]}}
_, observed, boot = boot_f1(y, three('roberta_syn_s{s}') + three('roberta_narr_s{s}')
                            + three('syn_w2vview_s{s}') + three('narr_w2vview_s{s}'), 400)
observed = observed.reshape(4, 3)
boot = boot.reshape(B, 4, 3)
per_training = (observed[0] - observed[1]) - (observed[2] - observed[3])
distribution = ((boot[:, 0] - boot[:, 1]) - (boot[:, 2] - boot[:, 3])).mean(1)
versus['roberta_delta_minus_bilstm_delta'] = {
    'mean': r6(per_training.mean()), 'ci95': [r6(v) for v in np.percentile(distribution, [2.5, 97.5])],
    'per_training': [r6(v) for v in per_training], 'B': B, 'rng_seed': SEED0 + 400}
out['roberta_vs_bilstm'] = versus

dual_records = {r['acn']: r for r in records if r.get('r2')}
first = np.load(os.path.join(HERE, 'dualpreds_w2vview_s0.npz'))
dual_acns = first['acns']
y_dual = first['y']
primary_preds = {}
second_preds = {}
for emb in ('w2vview', 'glove200'):
    for s in range(3):
        data = np.load(os.path.join(HERE, f'dualpreds_{emb}_s{s}.npz'))
        assert np.array_equal(data['acns'], dual_acns) and np.array_equal(data['y'], y_dual)
        primary_preds[(emb, s)] = (data['p1'] >= 0.5).astype(int)
        second_preds[(emb, s)] = (data['p2'] >= 0.5).astype(int)
# whitespace-token length ratio, as in seedavg_and_threshold.py
ratio = np.array([len(dual_records[a]['r2'].split()) / max(1, len(dual_records[a]['narr'].split())) for a in dual_acns])
second_tokens = np.array([len(TOKEN_RE.findall(dual_records[a]['r2'].lower())) for a in dual_acns])


def transfer(mask, seed):
    result = {}
    for emb in ('w2vview', 'glove200'):
        offset = 0 if emb == 'w2vview' else 50
        result[emb] = contrast(y_dual, [primary_preds[(emb, s)] for s in range(3)],
                               [second_preds[(emb, s)] for s in range(3)], seed + offset, mask)
    six_runs = result['w2vview']['per_training'] + result['glove200']['per_training']
    result['six_run_range'] = [min(six_runs), max(six_runs)]
    return result


out['dual_report_transfer_by_length'] = {
    'ratio_definition': 'whitespace-token length of supplemental / primary narrative (seedavg_and_threshold.py)',
    'bins': {'<0.25': int((ratio < 0.25).sum()), '0.25-0.5': int(((ratio >= 0.25) & (ratio < 0.5)).sum()),
             '0.5-0.75': int(((ratio >= 0.5) & (ratio < 0.75)).sum()),
             '0.75-1.0': int(((ratio >= 0.75) & (ratio < 1)).sum()), '>=1.0': int((ratio >= 1).sum())},
    'all_1530': transfer(np.ones(len(y_dual), bool), 3),
    'excluding_ratio_lt_0.25': transfer(ratio >= 0.25, 501),
    'ratio_lt_0.25_only': transfer(ratio < 0.25, 500),
    'excluding_supplemental_lt_25_tokens': transfer(second_tokens >= 25, 502),
    'class_balance_excluding_ratio_lt_0.25': {'aircraft': int(y_dual[ratio >= 0.25].sum()),
                                              'other': int((1 - y_dual[ratio >= 0.25]).sum())},
    'supplemental_tokens_median': {'ratio_lt_0.25': int(np.median(second_tokens[ratio < 0.25])),
                                   'ratio_ge_0.25': int(np.median(second_tokens[ratio >= 0.25]))}}

fields = ('summary', 'conseq', 'remedy')
y_shared = np.load(os.path.join(NHTSA, 'nhtsa_shared_preds_summary_bilstm_s0.npz'))['y']
y_field = np.load(os.path.join(NHTSA, 'nhtsa_preds_summary_bilstm_s0.npz'))['y']
shared = {f: [pred(os.path.join(NHTSA, f'nhtsa_shared_preds_{f}_bilstm_s{s}.npz')) for s in range(3)] for f in fields}
field_trained = {f: [pred(os.path.join(NHTSA, f'nhtsa_preds_{f}_bilstm_s{s}.npz')) for s in range(3)] for f in fields}
same_labels = True
for prefix in ('nhtsa_shared_preds', 'nhtsa_preds'):
    for f in fields:
        for s in range(3):
            labels = np.load(os.path.join(NHTSA, f'{prefix}_{f}_bilstm_s{s}.npz'))['y']
            same_labels = same_labels and bool(np.array_equal(labels, y_shared))
out['shared_embedding'] = {
    'asrs_shared_glove200_synopsis_minus_narrative': by_truncation['bilstm_glove200']['all_heldout'],
    'nhtsa_label_vectors_equal_shared_vs_field_trained': bool(np.array_equal(y_shared, y_field)),
    'nhtsa_all_files_same_labels': same_labels,
    'nhtsa_shared_word2vec': {
        'summary_minus_consequence': contrast(y_shared, shared['summary'], shared['conseq'], 601),
        'summary_minus_remedy': contrast(y_shared, shared['summary'], shared['remedy'], 602),
        'remedy_minus_consequence': contrast(y_shared, shared['remedy'], shared['conseq'], 603)},
    'nhtsa_field_trained_reproduction': {
        'summary_minus_consequence': contrast(y_field, field_trained['summary'], field_trained['conseq'], 6),
        'summary_minus_remedy': contrast(y_field, field_trained['summary'], field_trained['remedy'], 7),
        'remedy_minus_consequence': contrast(y_field, field_trained['remedy'], field_trained['conseq'], 604)},
    'stored_nhtsa_field_trained': {key: stored[key] for key in ('sum_conseq', 'sum_remedy')}}

means = {}
for name, pattern in (('tfidf_word', '{r}_tfidf'), ('tfidf_char', '{r}_charngram')):
    means[name] = {r: r6(f1_score(y, P[pattern.format(r=r)], average='macro')) for r in ('narr', 'syn')}
for name, pattern in (('bilstm_w2v', '{r}_w2vview_s{s}'), ('bilstm_glove200', '{r}_glove200_s{s}'),
                      ('meanpool_w2v', '{r}_meanmlp_s{s}'), ('meanpool_glove200', '{r}_meanmlpglove_s{s}'),
                      ('roberta', 'roberta_{r}_s{s}')):
    means[name] = {r: r6(np.mean([f1_score(y, P[pattern.format(r=r, s=s)], average='macro') for s in range(3)]))
                   for r in ('narr', 'syn')}
out['interaction_D_and_model_means'] = {
    'D_record_trained': reproduced['D_viewtrained'],
    'D_shared_glove200': interaction(y, three('syn_glove200_s{s}'), three('syn_meanmlpglove_s{s}'),
                                     three('narr_glove200_s{s}'), three('narr_meanmlpglove_s{s}'), 700),
    'three_training_means': means,
    'synopsis_minus_narrative': {key: {'mean': value['all_heldout']['mean'], 'ci95': value['all_heldout']['ci95'],
                                       'per_training': value['all_heldout']['per_training']}
                                 for key, value in by_truncation.items() if isinstance(value, dict)}}

draws = json.load(open(os.path.join(RESULTS, 'sensitivity_draws.json')))
extra = json.load(open(os.path.join(RESULTS, 'sensitivity_extra.json')))
per_draw = {d['draw']: [d['bilstm_delta']] for d in draws}
for d in sorted(extra, key=lambda d: (d['draw'], d['torch_seed'])):
    per_draw[d['draw']].append(d['bilstm_delta'])
all_runs = [v for values in per_draw.values() for v in values] + bilstm['per_training']
task_means = [np.mean(values) for values in per_draw.values()] + [bilstm['mean']]
out['redrawn_tasks_and_trainings'] = {
    'redrawn_task_bilstm_delta_per_draw': per_draw,
    'draw_means': {key: r6(np.mean(values)) for key, values in per_draw.items()},
    'main_task_per_training': bilstm['per_training'],
    'n_trainings': len(all_runs), 'n_positive': int(sum(v > 0 for v in all_runs)),
    'range': [r6(min(all_runs)), r6(max(all_runs))], 'median': r6(np.median(all_runs)),
    'mean_of_six_task_means': r6(np.mean(task_means))}

json.dump(out, open(os.path.join(RESULTS, 'truncation_and_transfer.json'), 'w'), indent=1)
print(json.dumps(out['reproduction_of_seedavg'], indent=1))
