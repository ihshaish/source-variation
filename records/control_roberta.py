"""Fine-tunes roberta-base per ASRS record type (narr cap 256, syn cap 64)
and per NHTSA field (cap 96), three seeds each, under one setup: AdamW
2e-5, batch 16, at most 3 epochs, early stop with patience 1 on a 5%
validation split, best weights, one held-out scoring. Reads
records_task.jsonl.gz and nhtsa/nhtsa_campaigns.jsonl; writes
roberta_preds_<key>_s<seed>.npz, control_roberta_partial.json after each
run, and control_roberta.json with the paired contrasts at the end.
Run: python3 control_roberta.py"""
import gzip
import json
import os
import re
from collections import Counter
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RECORDS = os.path.join(ROOT, 'records')
NHTSA = os.path.join(ROOT, 'nhtsa')
RESULTS = os.path.join(ROOT, 'results')
TOKEN_RE = re.compile(r"[a-z][a-z0-9/-]+")
SEED = 20260802
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained('roberta-base')


def log(*args):
    print(*args, flush=True)


def run(texts, y, is_test, cap, num_classes, key, seed):
    train_texts, val_texts, ytr, yva = train_test_split([t for t, m in zip(texts, ~is_test) if m], y[~is_test],
        test_size=0.05, stratify=y[~is_test], random_state=SEED + seed)
    torch.manual_seed(900 + seed)
    model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=num_classes).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-5)

    def encode(batch):
        return tokenizer(batch, truncation=True, max_length=cap, padding=True, return_tensors='pt')

    def predict(batch_texts):
        model.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(batch_texts), 64):
                x = encode(batch_texts[start:start + 64]).to(device)
                outputs.append(torch.softmax(model(**x).logits, 1).cpu().numpy())
        return np.concatenate(outputs)

    def macro_f1(yy, probs):
        return f1_score(yy, probs.argmax(1), average='macro')

    best, best_state, patience = -1, None, 0
    ytr_t = torch.from_numpy(np.asarray(ytr))
    for epoch in range(3):
        model.train()
        perm = torch.randperm(len(train_texts), generator=torch.Generator().manual_seed(seed * 77 + epoch))
        for start in range(0, len(perm), 16):
            batch = perm[start:start + 16].tolist()
            x = encode([train_texts[i] for i in batch]).to(device)
            loss = torch.nn.functional.cross_entropy(model(**x).logits, ytr_t[batch].to(device))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        f1 = macro_f1(yva, predict(val_texts))
        log(f"  {key} s{seed} epoch {epoch} val {f1:.4f}")
        if f1 > best + 1e-4:
            best, patience = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 1:
                break
    model.load_state_dict(best_state)
    probs = predict([t for t, m in zip(texts, is_test) if m])
    f1 = macro_f1(y[is_test], probs)
    np.savez(os.path.join(HERE, f'roberta_preds_{key}_s{seed}.npz'),
             probs=probs, pred=probs.argmax(1), y=y[is_test])
    del model
    if device == 'mps':
        torch.mps.empty_cache()
    return round(float(f1), 4)


results = []
rows = [json.loads(line) for line in open(os.path.join(NHTSA, 'nhtsa_campaigns.jsonl'))]
seen = set()
campaigns = []
for row in rows:
    if row['NHTSACampaignNumber'] in seen:
        continue
    seen.add(row['NHTSACampaignNumber'])
    campaigns.append(row)


def top_component(component):
    return (component or '').split(':')[0].split(',')[0].strip()


counts = Counter(top_component(row['Component']) for row in campaigns)
classes = sorted([k for k, v in counts.items() if v >= 300 and k])
label_index = {k: i for i, k in enumerate(classes)}
data = [row for row in campaigns if top_component(row['Component']) in label_index]
rng = np.random.default_rng(SEED)
parent = list(range(len(data)))


def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def union(a, b):
    root_a, root_b = find(a), find(b)
    if root_a != root_b:
        parent[root_b] = root_a


# campaigns sharing the first 400 characters of any text field stay on one side of the split
for field in ('Summary', 'Consequence', 'Remedy'):
    first = {}
    for i, row in enumerate(data):
        text_key = (row[field] or '').strip().lower()[:400]
        if not text_key:
            continue
        if text_key in first:
            union(first[text_key], i)
        else:
            first[text_key] = i
groups = {}
for i, row in enumerate(data):
    groups.setdefault(find(i), []).append(row)
group_keys = list(groups)
rng.shuffle(group_keys)
n_test = int(0.2 * len(data))
test = set()
count = 0
for group_key in group_keys:
    if count >= n_test:
        break
    for row in groups[group_key]:
        test.add(row['NHTSACampaignNumber'])
    count += len(groups[group_key])
for row in data:
    row['split'] = 'test' if row['NHTSACampaignNumber'] in test else 'train'
y_nhtsa = np.array([label_index[top_component(row['Component'])] for row in data])
test_nhtsa = np.array([row['split'] == 'test' for row in data])
log("NHTSA task:", len(data), "test", int(test_nhtsa.sum()))
for field_key, field in (('summary', 'Summary'), ('conseq', 'Consequence'), ('remedy', 'Remedy')):
    texts = [(row[field] or '') for row in data]
    for seed in (0, 1, 2):
        f1 = run(texts, y_nhtsa, test_nhtsa, 96, len(classes), f'nhtsa_{field_key}', seed)
        results.append({"key": f"roberta_nhtsa_{field_key}_s{seed}", "f1": f1})
        log(results[-1])
        json.dump(results, open(os.path.join(HERE, 'control_roberta_partial.json'), 'w'), indent=1)
records = [json.loads(line) for line in gzip.open(os.path.join(HERE, 'records_task.jsonl.gz'), 'rt')]
y_asrs = np.array([int(r['label']) for r in records])
test_asrs = np.array([r['split'] == 'test' for r in records])
log("ASRS task:", len(records), "test", int(test_asrs.sum()))
for record, cap in (('syn', 64), ('narr', 256)):
    texts = [r[record] for r in records]
    for seed in (0, 1, 2):
        f1 = run(texts, y_asrs, test_asrs, cap, 2, f'asrs_{record}', seed)
        results.append({"key": f"roberta_asrs_{record}_s{seed}", "f1": f1})
        log(results[-1])
        json.dump(results, open(os.path.join(HERE, 'control_roberta_partial.json'), 'w'), indent=1)

rng_paired = np.random.default_rng(SEED)


def macro_f1_pred(yy, pred):
    return f1_score(yy, pred, average='macro')


def paired(yy, pred_a, pred_b, n=5000):
    delta = macro_f1_pred(yy, pred_a) - macro_f1_pred(yy, pred_b)
    count = 0
    for _ in range(n):
        swap = rng_paired.random(len(yy)) < 0.5
        if abs(macro_f1_pred(yy, np.where(swap, pred_b, pred_a)) - macro_f1_pred(yy, np.where(swap, pred_a, pred_b))) >= abs(delta) - 1e-12:
            count += 1
    return round(float(delta), 4), round((count + 1) / (n + 1), 4)


contrasts = []
for seed in (0, 1, 2):
    syn = np.load(os.path.join(HERE, f'roberta_preds_asrs_syn_s{seed}.npz'))
    narr = np.load(os.path.join(HERE, f'roberta_preds_asrs_narr_s{seed}.npz'))
    delta, p = paired(syn['y'], syn['pred'], narr['pred'])
    contrasts.append({"contrast": f"roberta s{seed}: syn vs narr", "delta": delta, "p": p})
    log(contrasts[-1])
for seed in (0, 1, 2):
    preds = {field_key: np.load(os.path.join(HERE, f'roberta_preds_nhtsa_{field_key}_s{seed}.npz')) for field_key in ('summary', 'conseq', 'remedy')}
    for left, right in (('summary', 'conseq'), ('summary', 'remedy'), ('remedy', 'conseq')):
        delta, p = paired(preds[left]['y'], preds[left]['pred'], preds[right]['pred'])
        contrasts.append({"contrast": f"roberta nhtsa s{seed}: {left} vs {right}", "delta": delta, "p": p})
        log(contrasts[-1])
json.dump({"results": results, "contrasts": contrasts}, open(os.path.join(HERE, 'control_roberta.json'), 'w'), indent=1)
print("ROBERTA DONE", flush=True)
