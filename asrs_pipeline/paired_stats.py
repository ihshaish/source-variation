"""Computes the ASRS statistics from the saved held-out predictions.

Nothing here trains. For every configuration it gives a bootstrap 95% CI on
macro-F1 (10,000 record resamples), then runs paired approximate-randomisation
tests within four families, Holm-corrected inside each family: A, embeddings
against GloVe-200; B, BiLSTM against BiGRU; C, fastText against word2vec; D,
BiLSTM against the CNN and the mean-pooling MLP. Pairs are formed by record,
because every model was scored on the same held-out set.

Reads the seed-0 preds_*.npz files from ASRS_RES and writes asrs_stats.json
there. Run: python3 paired_stats.py
"""
import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("ASRS_RES", os.path.join(HERE, "results"))
B = 10_000
rng = np.random.default_rng(20260802)


def macro_f1(y, yhat):
    out = 0.0
    for label in (0, 1):
        tp = np.count_nonzero((yhat == label) & (y == label))
        fp = np.count_nonzero((yhat == label) & (y != label))
        fn = np.count_nonzero((yhat != label) & (y == label))
        denom = 2 * tp + fp + fn
        out += (2 * tp / denom) if denom else 0.0
    return out / 2


def load_preds():
    preds = {}
    for path in glob.glob(os.path.join(RES, "preds_*_s0.npz")):
        match = re.match(r"preds_(.+)_(bilstm|bigru|cnn|meanmlp)_s0\.npz", os.path.basename(path))
        if not match:
            continue
        saved = np.load(path)
        preds[(match.group(1), match.group(2))] = ((saved["probs"] >= 0.5).astype(int), saved["y"])
    return preds


def boot_ci(yhat, y):
    n = len(y)
    stats = np.empty(B)
    for b in range(B):
        sample = rng.integers(0, n, n)
        stats[b] = macro_f1(y[sample], yhat[sample])
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def paired_randomisation(preds_a, preds_b, y):
    observed = abs(macro_f1(y, preds_a) - macro_f1(y, preds_b))
    differs = preds_a != preds_b
    n = len(y)
    count = 0
    for _ in range(B):
        swap = rng.random(n) < 0.5
        swapped = swap & differs
        a_swapped = np.where(swapped, preds_b, preds_a)
        b_swapped = np.where(swapped, preds_a, preds_b)
        if abs(macro_f1(y, a_swapped) - macro_f1(y, b_swapped)) >= observed - 1e-12:
            count += 1
    return (count + 1) / (B + 1)


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adjusted[i] = running
    return adjusted


def main():
    preds = load_preds()
    report = {"configs": {}, "contrasts": []}
    for (emb, arch), (yhat, y) in sorted(preds.items()):
        f1 = macro_f1(y, yhat)
        lo, hi = boot_ci(yhat, y)
        report["configs"][f"{emb}_{arch}"] = {
            "test_macro_f1": round(float(f1), 4), "ci95": [round(lo, 4), round(hi, 4)]}
        print(f"{emb}/{arch}: {f1:.4f} [{lo:.4f}, {hi:.4f}]")

    embs = sorted({e for e, _ in preds})
    families = []
    for arch in ("bilstm", "bigru"):
        families.append([("A-emb", arch, "glove200", other) for other in embs
                         if other != "glove200"
                         and (other, arch) in preds and ("glove200", arch) in preds])
    families.append([("B-arch", emb, "bilstm", "bigru") for emb in embs
                     if (emb, "bilstm") in preds and (emb, "bigru") in preds])
    families.append([("C-subword", arch, "fasttext", "w2vasrs") for arch in ("bilstm", "bigru")
                     if ("fasttext", arch) in preds and ("w2vasrs", arch) in preds])
    families.append([("D-sequence", "glove200", "bilstm", probe) for probe in ("cnn", "meanmlp")
                     if ("glove200", probe) in preds and ("glove200", "bilstm") in preds])

    for family in families:
        if not family:
            continue
        pvalues = []
        rows = []
        for tag, fixed, first, second in family:
            # in families A and C the fixed part is the architecture; in B and D it is the embedding
            if tag in ("A-emb", "C-subword"):
                preds_a, y = preds[(first, fixed)]
                preds_b, _ = preds[(second, fixed)]
            else:
                preds_a, y = preds[(fixed, first)]
                preds_b, _ = preds[(fixed, second)]
            p = paired_randomisation(preds_a, preds_b, y)
            delta = macro_f1(y, preds_a) - macro_f1(y, preds_b)
            pvalues.append(p)
            rows.append({"family": tag, "contrast": f"{fixed}: {first} vs {second}",
                         "delta_f1": round(float(delta), 4), "p": p})
        for row, p_adjusted in zip(rows, holm(pvalues)):
            row["p_holm"] = round(p_adjusted, 4)
            row["p"] = round(row["p"], 4)
            report["contrasts"].append(row)
            print(row)

    with open(os.path.join(RES, "asrs_stats.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
