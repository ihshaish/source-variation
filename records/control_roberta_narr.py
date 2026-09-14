"""Runs one ASRS narrative RoBERTa fine-tune per process; twelve runs in
one process exhausted a 16 GB machine into swap, so the narrative runs of
control_roberta.py are taken one at a time here. Same setup as
control_roberta.py: AdamW 2e-5, batch 16, at most 3 epochs, early stop
with patience 1 on a 5% validation split, cap 256, one held-out scoring.
Reads records_task.jsonl.gz and control_roberta_partial.json; writes
roberta_preds_asrs_narr_s<seed>.npz and appends the result to
control_roberta_partial.json.
Run: python3 control_roberta_narr.py <seed>"""
import gzip
import json
import os
import sys
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
SEED = 20260802
seed = int(sys.argv[1])
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
records = [json.loads(line) for line in gzip.open(os.path.join(HERE, 'records_task.jsonl.gz'), 'rt')]
y = np.array([int(r['label']) for r in records])
is_test = np.array([r['split'] == 'test' for r in records])
texts = [r['narr'] for r in records]
cap = 256
print(f"narr s{seed}: task {len(records)} test {int(is_test.sum())}", flush=True)
train_texts, val_texts, ytr, yva = train_test_split([t for t, m in zip(texts, ~is_test) if m], y[~is_test],
    test_size=0.05, stratify=y[~is_test], random_state=SEED + seed)
torch.manual_seed(900 + seed)
model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=2).to(device)
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
    print(f"  asrs_narr s{seed} epoch {epoch} val {f1:.4f}", flush=True)
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
np.savez(os.path.join(HERE, f'roberta_preds_asrs_narr_s{seed}.npz'),
         probs=probs, pred=probs.argmax(1), y=y[is_test])
results = json.load(open(os.path.join(HERE, 'control_roberta_partial.json')))
results.append({"key": f"roberta_asrs_narr_s{seed}", "f1": round(float(f1), 4)})
json.dump(results, open(os.path.join(HERE, 'control_roberta_partial.json'), 'w'), indent=1)
print(json.dumps(results[-1]), flush=True)
