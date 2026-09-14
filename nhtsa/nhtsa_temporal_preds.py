"""Reruns nhtsa_temporal.py with the same task, seeds and models, keeping the
per-record test predictions so that the field contrasts can be tested.
The BiLSTM family is summary against consequence and summary against remedy
in each of the three trainings, six contrasts with Holm correction; the two
TF-IDF contrasts are reported without correction. Each contrast is a paired
randomisation test that swaps the two predictions per record at random,
10000 draws, p = (count + 1) / (n + 1). Writes nhtsa_temporal_preds.npz and
results/nhtsa/nhtsa_temporal_tests.json. Run: python3 nhtsa_temporal_preds.py"""
import json
import os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RECORDS = os.path.join(ROOT, 'records')
NHTSA = os.path.join(ROOT, 'nhtsa')
RESULTS = os.path.join(ROOT, 'results')
import sys
sys.path.insert(0, HERE)
from nhtsa_temporal import build, TOKEN_RE, PAD, OOV, CAP


def macro_f1(y, pred):
    from sklearn.metrics import f1_score
    return f1_score(y, pred, average='macro')


def paired(y, pred_a, pred_b, rng, n=10000):
    d0 = macro_f1(y, pred_a) - macro_f1(y, pred_b)
    count = 0
    for _ in range(n):
        swap = rng.random(len(y)) < 0.5
        swapped_a = np.where(swap, pred_b, pred_a)
        swapped_b = np.where(swap, pred_a, pred_b)
        if abs(macro_f1(y, swapped_a) - macro_f1(y, swapped_b)) >= abs(d0) - 1e-12:
            count += 1
    return d0, (count + 1) / (n + 1)


def main():
    import torch
    import torch.nn as nn
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from gensim.models import Word2Vec
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    data, y, is_test, n_classes = build()
    predictions = {"y": y[is_test]}

    class RNN(nn.Module):
        def __init__(self, matrix, n_classes):
            super().__init__()
            self.emb = nn.Embedding.from_pretrained(torch.from_numpy(matrix), freeze=True, padding_idx=PAD)
            self.rnn = nn.LSTM(matrix.shape[1], 64, batch_first=True, bidirectional=True)
            self.drop = nn.Dropout(0.3)
            self.fc = nn.Linear(128, n_classes)

        def forward(self, x):
            hidden = self.rnn(self.emb(x))[1][0]
            return self.fc(self.drop(torch.cat([hidden[0], hidden[1]], dim=1)))

    def predict(model, X):
        model.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(X), 256):
                outputs.append(model(torch.from_numpy(X[start:start + 256]).long().to(device)).argmax(1).cpu().numpy())
        return np.concatenate(outputs)

    for field_key, field in (('summary', 'Summary'), ('conseq', 'Consequence'), ('remedy', 'Remedy')):
        texts = [(record[field] or '') for record in data]
        vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
        Xtr = vectorizer.fit_transform([t for t, m in zip(texts, ~is_test) if m])
        Xte = vectorizer.transform([t for t, m in zip(texts, is_test) if m])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, y[~is_test])
        predictions[f'tfidf_{field_key}'] = clf.predict(Xte)
        print('tfidf', field_key, round(macro_f1(y[is_test], predictions[f'tfidf_{field_key}']), 4), flush=True)
        sentences = [TOKEN_RE.findall(t.lower()) for t, m in zip(texts, ~is_test) if m and t.strip()]
        w2v = Word2Vec(vector_size=200, window=5, min_count=3, sg=1, epochs=10, workers=8, seed=20260802)
        w2v.build_vocab(sentences)
        w2v.train(corpus_iterable=sentences, total_examples=len(sentences), epochs=10)
        token_lists = [TOKEN_RE.findall(t.lower())[:CAP] for t in texts]
        vocab = set(t for ts in token_lists for t in ts)
        oov_rng = np.random.default_rng(0)
        word_index, vectors = {}, [np.zeros(200, np.float32), oov_rng.normal(0, 0.1, 200).astype(np.float32)]
        for word in sorted(vocab):
            if word in w2v.wv:
                word_index[word] = len(vectors)
                vectors.append(w2v.wv[word].astype(np.float32))
        matrix = np.stack(vectors)
        del w2v
        X = np.zeros((len(token_lists), CAP), np.int32)
        for i, ts in enumerate(token_lists):
            for j, t in enumerate(ts):
                X[i, j] = word_index.get(t, OOV)
        Xtr_all, ytr_all = X[~is_test], y[~is_test]
        Xte_, yte_ = X[is_test], y[is_test]
        from sklearn.metrics import f1_score
        import torch.nn as nn
        loss_fn = nn.CrossEntropyLoss()
        for seed in (0, 1, 2):
            Xtr2, Xva, ytr2, yva = train_test_split(Xtr_all, ytr_all, test_size=0.05, stratify=ytr_all, random_state=20260802 + seed)
            torch.manual_seed(100 + seed)
            model = RNN(matrix, n_classes).to(device)
            optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
            Xtr_t = torch.from_numpy(Xtr2).long()
            ytr_t = torch.from_numpy(ytr2)
            best_f1, best_state, patience = -1.0, None, 0
            for epoch in range(15):
                model.train()
                perm = torch.randperm(len(Xtr_t), generator=torch.Generator().manual_seed((100 + seed) * 1000 + epoch))
                for start in range(0, len(perm), 128):
                    batch_idx = perm[start:start + 128]
                    optimizer.zero_grad()
                    loss = loss_fn(model(Xtr_t[batch_idx].to(device)), ytr_t[batch_idx].to(device))
                    loss.backward()
                    optimizer.step()
                f1 = f1_score(yva, predict(model, Xva), average='macro')
                if f1 > best_f1 + 1e-4:
                    best_f1, patience = f1, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience += 1
                    if patience >= 2:
                        break
            if best_state:
                model.load_state_dict(best_state)
            predictions[f'bilstm_{field_key}_s{seed}'] = predict(model, Xte_)
            print('bilstm', field_key, seed, round(macro_f1(yte_, predictions[f'bilstm_{field_key}_s{seed}']), 4), flush=True)
            del model
        del matrix, X, Xtr_t
    np.savez(os.path.join(HERE, 'nhtsa_temporal_preds.npz'), **predictions)
    rng = np.random.default_rng(20260802)
    family = []
    for seed in (0, 1, 2):
        for field_a, field_b in (('summary', 'conseq'), ('summary', 'remedy')):
            d0, p = paired(predictions['y'], predictions[f'bilstm_{field_a}_s{seed}'], predictions[f'bilstm_{field_b}_s{seed}'], rng)
            family.append({"contrast": f"bilstm s{seed}: {field_a}-{field_b}", "delta": round(float(d0), 4), "p": p})
    order = np.argsort([contrast['p'] for contrast in family])
    n_contrasts = len(family)
    running = 0.0
    for rank, i in enumerate(order):
        adjusted = min(1.0, (n_contrasts - rank) * family[i]['p'])
        running = max(running, adjusted)
        family[i]['p_holm'] = round(running, 4)
        family[i]['p'] = round(family[i]['p'], 5)
    descriptive = []
    for field_a, field_b in (('summary', 'conseq'), ('summary', 'remedy')):
        d0, p = paired(predictions['y'], predictions[f'tfidf_{field_a}'], predictions[f'tfidf_{field_b}'], rng)
        descriptive.append({"contrast": f"tfidf: {field_a}-{field_b}", "delta": round(float(d0), 4), "p": round(p, 5)})
    scores = {k: round(float(macro_f1(predictions['y'], predictions[k])), 4) for k in predictions if k != 'y'}
    json.dump({"n_test": int(len(predictions['y'])), "scores": scores, "family_holm": family, "tfidf_descriptive": descriptive},
              open(os.path.join(RESULTS, 'nhtsa', 'nhtsa_temporal_tests.json'), 'w'), indent=1)
    print("TEMPORAL TESTS DONE", flush=True)


if __name__ == '__main__':
    main()
