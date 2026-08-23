"""The statistics, all of them, from saved held-out predictions.

Nothing here trains. Bootstrap CIs (10k record resamples) per configuration,
then paired approximate-randomisation tests for the pre-specified families,
Holm-corrected within family: A embeddings, B architectures, C subword
(fastText vs word2vec), D sequence (BiLSTM vs CNN / mean pooling). Paired by
record -- every model saw the identical held-out set.

python3 l1_stats.py   writes results/l1_stats.json, which is the file the
paper's ASRS numbers come from.
"""
import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("L1_RES", os.path.join(HERE, "results"))
B = 10_000
rng = np.random.default_rng(20260802)


def macro_f1(y, yhat):
    """Binary macro-F1 via counts (fast; no sklearn overhead)."""
    out = 0.0
    for c in (0, 1):
        tp = np.count_nonzero((yhat == c) & (y == c))
        fp = np.count_nonzero((yhat == c) & (y != c))
        fn = np.count_nonzero((yhat != c) & (y == c))
        denom = 2 * tp + fp + fn
        out += (2 * tp / denom) if denom else 0.0
    return out / 2


def load_preds():
    out = {}
    for path in glob.glob(os.path.join(RES, "preds_*_s0.npz")):
        m = re.match(r"preds_(.+)_(bilstm|bigru|cnn|meanmlp)_s0\.npz", os.path.basename(path))
        if not m:
            continue
        d = np.load(path)
        out[(m.group(1), m.group(2))] = ((d["probs"] >= 0.5).astype(int), d["y"])
    return out


def boot_ci(yhat, y):
    n = len(y)
    stats = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        stats[b] = macro_f1(y[i], yhat[i])
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def paired_randomisation(a, b, y):
    obs = abs(macro_f1(y, a) - macro_f1(y, b))
    diff = a != b
    n = len(y)
    count = 0
    for _ in range(B):
        swap = rng.random(n) < 0.5
        s = swap & diff
        aa = np.where(s, b, a)
        bb = np.where(s, a, b)
        if abs(macro_f1(y, aa) - macro_f1(y, bb)) >= obs - 1e-12:
            count += 1
    return (count + 1) / (B + 1)


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


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

    for fam in families:
        if not fam:
            continue
        ps, rows = [], []
        for tag, fixed, s1, s2 in fam:
            if tag in ("A-emb", "C-subword"):
                a, y = preds[(s1, fixed)]
                b, _ = preds[(s2, fixed)]
            else:  # B-arch, D-sequence: fixed = embedding, s1/s2 = architectures
                a, y = preds[(fixed, s1)]
                b, _ = preds[(fixed, s2)]
            p = paired_randomisation(a, b, y)
            d = macro_f1(y, a) - macro_f1(y, b)
            ps.append(p)
            rows.append({"family": tag, "contrast": f"{fixed}: {s1} vs {s2}",
                         "delta_f1": round(float(d), 4), "p": p})
        for row, padj in zip(rows, holm(ps)):
            row["p_holm"] = round(padj, 4)
            row["p"] = round(row["p"], 4)
            report["contrasts"].append(row)
            print(row)

    with open(os.path.join(RES, "l1_stats.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
