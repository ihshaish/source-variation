"""Trains the final BiLSTM for one record type and one embedding, three
seeds, on the paper's task and split, and saves the held-out predictions
with one result line per seed. Narrative models are also scored on the
dual-report subset reading the second reporter's account, paired by
record. Reads records_task.jsonl.gz and either glove.6B.200d.txt from
EMB_DIR or w2v_<record>_200d.kv; writes preds_<record>_<emb>_s<seed>.npz,
dualpreds_<emb>_s<seed>.npz and appends to records_results.jsonl.
Run: python3 records_train.py --record narr|syn --emb glove200|w2vview [--smoke]"""
import argparse
import gzip
import json
import os
import re
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors

HERE = os.path.dirname(os.path.abspath(__file__))
TOKEN_RE = re.compile(r"[a-z][a-z0-9/-]+")
SEED = 20260802
PAD, OOV = 0, 1
CAPS = {'narr': 256, 'syn': 64, 'r2': 256}
GLOVE = os.path.join(os.environ.get('EMB_DIR', '.'), 'glove.6B.200d.txt')


def tokenise(text, cap):
    return TOKEN_RE.findall(text.lower())[:cap]


def load(record):
    records = []
    with gzip.open(os.path.join(HERE, 'records_task.jsonl.gz'), 'rt') as f:
        for line in f:
            row = json.loads(line)
            row['toks'] = tokenise(row[record], CAPS[record])
            row['r2toks'] = tokenise(row['r2'], CAPS['r2']) if record == 'narr' and len(row['r2']) > 40 else None
            records.append(row)
    return records


def build_matrix(emb, record, vocab):
    rng = np.random.default_rng(0)
    if emb == 'glove200':
        index = {}
        vectors = [np.zeros(200, np.float32), rng.normal(0, 0.1, 200).astype(np.float32)]
        for line in open(GLOVE, encoding='utf-8'):
            parts = line.rstrip().split(' ')
            if parts[0] in vocab:
                index[parts[0]] = len(vectors)
                vectors.append(np.asarray(parts[1:], np.float32))
        return index, np.stack(vectors)
    keyed = KeyedVectors.load(os.path.join(HERE, f"w2v_{record}_200d.kv"), mmap='r')
    index = {}
    vectors = [np.zeros(200, np.float32), rng.normal(0, 0.1, 200).astype(np.float32)]
    for word in sorted(vocab):
        if word in keyed:
            index[word] = len(vectors)
            vectors.append(keyed[word].astype(np.float32))
    return index, np.stack(vectors)


def encode(token_lists, index, cap):
    X = np.zeros((len(token_lists), cap), np.int32)
    for i, tokens in enumerate(token_lists):
        for j, token in enumerate(tokens):
            X[i, j] = index.get(token, OOV)
    return X


class RNN(nn.Module):
    def __init__(self, matrix):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(torch.from_numpy(matrix), freeze=True, padding_idx=PAD)
        self.rnn = nn.LSTM(matrix.shape[1], 64, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        hidden = self.rnn(self.emb(x))[1][0]
        return self.fc(self.drop(torch.cat([hidden[0], hidden[1]], dim=1)))


def predict(model, X, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(X), 256):
            outputs.append(torch.softmax(model(torch.from_numpy(X[start:start + 256]).long().to(device)), 1)[:, 1].cpu().numpy())
    return np.concatenate(outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', required=True, choices=['narr', 'syn'])
    parser.add_argument('--emb', required=True, choices=['glove200', 'w2vview'])
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    records = load(args.record)
    if args.smoke:
        rng = np.random.default_rng(0)
        records = [records[i] for i in rng.choice(len(records), 3000, replace=False)]
    vocab = set(t for r in records for t in r['toks'])
    if args.record == 'narr':
        vocab |= set(t for r in records if r['r2toks'] for t in r['r2toks'])
    index, matrix = build_matrix(args.emb, args.record, vocab)
    cap = CAPS[args.record]
    X = encode([r['toks'] for r in records], index, cap)
    y = np.array([int(r['label']) for r in records], np.int64)
    is_test = np.array([r['split'] == 'test' for r in records])
    acns = np.array([r['acn'] for r in records])
    Xtr_all, ytr_all = X[~is_test], y[~is_test]
    Xte, yte = X[is_test], y[is_test]
    dual = None
    if args.record == 'narr':
        dual_index = [i for i in range(len(records)) if is_test[i] and records[i]['r2toks']]
        Xr2 = encode([records[i]['r2toks'] for i in dual_index], index, CAPS['r2'])
        Xr1 = X[dual_index]
        yd = y[dual_index]
        dual_acns = acns[dual_index]
        dual = (Xr1, Xr2, yd, dual_acns)
        print(f"dual-report held-out records: {len(dual_index)}")
    tag = 'smoke_' if args.smoke else ''
    done = set()
    results_path = os.path.join(HERE, 'records_results.jsonl')
    if os.path.exists(results_path):
        done = {json.loads(line)['key'] for line in open(results_path)}
    for seed in ((0,) if args.smoke else (0, 1, 2)):
        key = f"{tag}final_{args.record}_{args.emb}_s{seed}"
        if key in done:
            print("skip", key)
            continue
        Xtr, Xva, ytr, yva = train_test_split(Xtr_all, ytr_all, test_size=0.05, stratify=ytr_all, random_state=SEED + seed)
        torch.manual_seed(100 + seed)
        model = RNN(matrix).to(device)
        optimiser = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        Xtr_t = torch.from_numpy(Xtr).long()
        ytr_t = torch.from_numpy(ytr)
        best, best_state, patience = -1.0, None, 0
        start_time = time.time()
        for epoch in range(1 if args.smoke else 15):
            model.train()
            perm = torch.randperm(len(Xtr_t), generator=torch.Generator().manual_seed((100 + seed) * 1000 + epoch))
            for start in range(0, len(perm), 128):
                batch = perm[start:start + 128]
                optimiser.zero_grad()
                loss = loss_fn(model(Xtr_t[batch].to(device)), ytr_t[batch].to(device))
                loss.backward()
                optimiser.step()
            f1 = f1_score(yva, (predict(model, Xva, device) >= 0.5).astype(int), average='macro')
            if f1 > best + 1e-4:
                best, patience = f1, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= 2:
                    break
        if best_state:
            model.load_state_dict(best_state)
        test_probs = predict(model, Xte, device)
        f1 = f1_score(yte, (test_probs >= 0.5).astype(int), average='macro')
        np.savez(os.path.join(HERE, f"{tag}preds_{args.record}_{args.emb}_s{seed}.npz"), probs=test_probs, y=yte, acns=acns[is_test])
        out = {"key": key, "view": args.record, "emb": args.emb, "seed": seed, "test_macro_f1": round(float(f1), 4),
               "val_f1": round(float(best), 4), "secs": round(time.time() - start_time)}
        if dual is not None:
            Xr1, Xr2, yd, dual_acns = dual
            probs_r1 = predict(model, Xr1, device)
            probs_r2 = predict(model, Xr2, device)
            np.savez(os.path.join(HERE, f"{tag}dualpreds_{args.emb}_s{seed}.npz"), p1=probs_r1, p2=probs_r2, y=yd, acns=dual_acns)
            out["dual_r1_f1"] = round(float(f1_score(yd, (probs_r1 >= 0.5).astype(int), average='macro')), 4)
            out["dual_r2_f1"] = round(float(f1_score(yd, (probs_r2 >= 0.5).astype(int), average='macro')), 4)
        with open(results_path, 'a') as f:
            f.write(json.dumps(out) + '\n')
        print(json.dumps(out))


if __name__ == '__main__':
    main()
