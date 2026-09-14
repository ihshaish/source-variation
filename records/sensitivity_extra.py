"""Adds BiLSTM trainings two and three (torch seeds 101 and 102) to each
negative-class redraw, with the same draw, split and setup as
sensitivity_draws.py. Reads pool_records.jsonl.gz and appends one result
per run to sensitivity_extra.json.
Run: python3 sensitivity_extra.py <draw 1..5> <torch seed>"""
import gzip
import json
import os
import re
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RECORDS = os.path.join(ROOT, 'records')
NHTSA = os.path.join(ROOT, 'nhtsa')
RESULTS = os.path.join(ROOT, 'results')
POOL = os.path.join(HERE, 'pool_records.jsonl.gz')
TOKEN_RE = re.compile(r"[a-z][a-z0-9/-]+")
PAD, OOV = 0, 1
CAPS = {'narr': 256, 'syn': 64}


def main(draw, torch_seed):
    import torch
    import torch.nn as nn
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    from gensim.models import Word2Vec
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    records = [json.loads(line) for line in gzip.open(POOL, 'rt')]
    positives = [r for r in records if r['label'] == 1]
    negatives = [r for r in records if r['label'] == 0]
    rng = np.random.default_rng(100 + draw)
    keep = rng.choice(len(negatives), size=len(positives), replace=False)
    data = positives + [negatives[i] for i in sorted(keep)]
    order = rng.permutation(len(data))
    data = [data[i] for i in order]
    test = set()
    for label in (0, 1):
        class_acns = [r['acn'] for r in data if r['label'] == label]
        class_order = rng.permutation(len(class_acns))
        n_test = round(0.2 * len(class_acns))
        test.update(class_acns[i] for i in class_order[:n_test])
    for r in data:
        r['split'] = 'test' if r['acn'] in test else 'train'
    y = np.array([r['label'] for r in data], np.int64)
    is_test = np.array([r['split'] == 'test' for r in data])
    result = {"draw": draw, "torch_seed": torch_seed}

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

    def predict(model, X):
        model.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(X), 256):
                outputs.append(torch.softmax(model(torch.from_numpy(X[start:start + 256]).long().to(device)), 1)[:, 1].cpu().numpy())
        return np.concatenate(outputs)

    for record in ('narr', 'syn'):
        cap = CAPS[record]
        sentences = [TOKEN_RE.findall(r[record].lower()) for r in data if r['split'] == 'train' and r[record].strip()]
        w2v = Word2Vec(vector_size=200, window=5, min_count=3, sg=1, epochs=10, workers=8, seed=20260802)
        w2v.build_vocab(sentences)
        w2v.train(corpus_iterable=sentences, total_examples=len(sentences), epochs=10)
        token_lists = [TOKEN_RE.findall(r[record].lower())[:cap] for r in data]
        vocab = set(t for ts in token_lists for t in ts)
        init_rng = np.random.default_rng(0)
        index = {}
        vectors = [np.zeros(200, np.float32), init_rng.normal(0, 0.1, 200).astype(np.float32)]
        for word in sorted(vocab):
            if word in w2v.wv:
                index[word] = len(vectors)
                vectors.append(w2v.wv[word].astype(np.float32))
        matrix = np.stack(vectors)
        del w2v
        X = np.zeros((len(token_lists), cap), np.int32)
        for i, tokens in enumerate(token_lists):
            for j, token in enumerate(tokens):
                X[i, j] = index.get(token, OOV)
        Xtr_all, ytr_all = X[~is_test], y[~is_test]
        X_test, y_test = X[is_test], y[is_test]
        Xtr, Xva, ytr, yva = train_test_split(Xtr_all, ytr_all, test_size=0.05, stratify=ytr_all, random_state=20260802)
        torch.manual_seed(torch_seed)
        model = RNN(matrix).to(device)
        optimiser = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        Xtr_t = torch.from_numpy(Xtr).long()
        ytr_t = torch.from_numpy(ytr)
        best, best_state, patience = -1.0, None, 0
        for epoch in range(15):
            model.train()
            perm = torch.randperm(len(Xtr_t), generator=torch.Generator().manual_seed(torch_seed * 1000 + epoch))
            for start in range(0, len(perm), 128):
                batch = perm[start:start + 128]
                optimiser.zero_grad()
                loss = loss_fn(model(Xtr_t[batch].to(device)), ytr_t[batch].to(device))
                loss.backward()
                optimiser.step()
            from sklearn.metrics import f1_score as f1s
            f1 = f1s(yva, (predict(model, Xva) >= 0.5).astype(int), average='macro')
            print(f"  d{draw} t{torch_seed} {record} epoch {epoch} val {f1:.4f}", flush=True)
            if f1 > best + 1e-4:
                best, patience = f1, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= 2:
                    break
        if best_state:
            model.load_state_dict(best_state)
        from sklearn.metrics import f1_score as f1s
        pred = (predict(model, X_test) >= 0.5).astype(int)
        result[f'bilstm_{record}'] = round(float(f1s(y_test, pred, average='macro')), 4)
        del model, matrix, X, Xtr_t
    result['bilstm_delta'] = round(result['bilstm_syn'] - result['bilstm_narr'], 4)
    results_path = os.path.join(HERE, 'sensitivity_extra.json')
    results = json.load(open(results_path)) if os.path.exists(results_path) else []
    results.append(result)
    json.dump(results, open(results_path, 'w'), indent=1)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]))
