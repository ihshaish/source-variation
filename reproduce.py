"""This notebook will print paper's ASRS numbers next to what shows in results/.

No training or downloads required - so this just reads the two results files and checks
that every headline claim in the article matches them. Run it after cloning:

    python3 reproduce.py

Each line here will essentially show the paper's value, the shipped value, and an OK or MISMATCH.
If you ever see MISMATCH, something has drifted - please tell us:
- hisham.ihshaish@uwe.ac.uk (Hisham)
- peter.mayhew@geaerospace.com (Peter)
"""
import json, os

here = os.path.dirname(os.path.abspath(__file__))
stats = json.load(open(os.path.join(here, "results", "l1_stats.json")))
cfg, con = stats["configs"], stats["contrasts"]

checks = []
def claim(label, paper, shipped, tol=0.0006):
    checks.append((label, paper, shipped, abs(paper - shipped) <= tol))

# held-out macro-F1 (paper Table 6 / Supplementary Table S4)
for key, val in [("glove200_bilstm", 0.860), ("glove300_bilstm", 0.862),
                 ("w2vasrs_bilstm", 0.881), ("fasttext_bigru", 0.879),
                 ("w2vasrs_bigru", 0.880), ("glove200_cnn", 0.869),
                 ("glove200_meanmlp", 0.836), ("glove50_bilstm", 0.829)]:
    claim(f"held-out {key}", val, cfg[key]["test_macro_f1"])

# paired contrasts the text leans on (Supplementary Table S3)
def find(family, name):
    return next(c for c in con if c["family"] == family and c["contrast"] == name)
c = find("A-emb", "bilstm: glove200 vs w2vasrs")
claim("in-domain gain (BiLSTM, w2v vs glove200)", -0.020, c["delta_f1"], 0.0006)
assert c["p_holm"] < 0.001, "w2v gain should be significant"
c = find("C-subword", "bilstm: fasttext vs w2vasrs")
claim("subword vs word-level (BiLSTM)", -0.015, c["delta_f1"], 0.0006)
c = find("C-subword", "bigru: fasttext vs w2vasrs")
assert abs(c["delta_f1"]) <= 0.005 and c["p_holm"] > 0.05, "BiGRU subword null"
c = find("B-arch", "w2vasrs: bilstm vs bigru")
assert abs(c["delta_f1"]) <= 0.002 and c["p_holm"] > 0.05, "cell gap gone at best embedding"
c = find("B-arch", "glove50: bilstm vs bigru")
claim("cell gap at weakest embedding", -0.032, c["delta_f1"], 0.0006)
c = find("D-sequence", "glove200: bilstm vs meanmlp")
claim("pooling cost (ASRS)", 0.025, c["delta_f1"], 0.0006)

# order and cap probes (paper section on boundary conditions)
abl = [json.loads(l) for l in open(os.path.join(here, "results", "ablation_results.jsonl"))]
shuf = [r["test_macro_f1"] for r in abl if r["key"].startswith("shuffle_final")]
base = 0.859  # three-run mean of the unshuffled BiLSTM/GloVe-200 finals
worst = max(abs(base - s) for s in shuf)
checks.append(("order shuffle cost <= 0.005 (paper)", 0.005, round(worst, 4), worst <= 0.005))
cap = [r["test_macro_f1"] for r in abl if r["key"].startswith("cap512_final")]
worstc = max(abs(0.860 - cv) for cv in cap)
checks.append(("512-cap change <= 0.007 (paper)", 0.007, round(worstc, 4), worstc <= 0.007))

bad = 0
for label, p, s, ok in checks:
    print(f"{'OK      ' if ok else 'MISMATCH'} {label}: paper {p}  shipped {round(s,4)}")
    bad += 0 if ok else 1
print(f"\n{len(checks)-bad}/{len(checks)} checks pass")
raise SystemExit(1 if bad else 0)
