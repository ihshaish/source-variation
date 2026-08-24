"""The script below will print paper's ASRS numbers next to what shows in results/.

No training or downloads required - so this just reads the two results files and checks
that every headline claim in the article matches them. Run it after cloning:

    python3 reproduce.py

Each line here will essentially show the paper's value, the reported value, & an OK or MISMATCH.
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

# --- matched views (paper section "Record views on public safety reports") ---
v = json.load(open(os.path.join(here, "results", "views", "views_stats.json")))
seq = [c for c in v["contrasts"] if c["family"] == "view" and "tfidf" not in c["contrast"]]
ds = [-c["delta_f1"] for c in seq]  # stored narr-syn; paper reports syn advantage
checks.append(("synopsis > narrative, all 6 sequence runs in 0.011-0.021",
               "0.011-0.021", f"{min(ds)}-{max(ds)}",
               len(seq) == 6 and all(0.011 <= round(d, 3) <= 0.021 for d in ds)
               and all(c["p_holm"] <= 0.0013 for c in seq)))
tf = next(c for c in v["contrasts"] if "tfidf" in c["contrast"])
checks.append(("TF-IDF view contrast null (p=0.78)", 0.78, tf["p"], abs(tf["p"] - 0.7806) < 0.01))
em = json.load(open(os.path.join(here, "results", "views", "echo_mask.json")))
checks.append(("masked synopsis still beats plain narrative", 0.8711, em["syn"]["echo_masked"],
               em["syn"]["echo_masked"] > em["narr"]["plain"]))
dual = [c for c in v["contrasts"] if c["family"] == "author"]
dd = [c["delta_f1"] for c in dual]
checks.append(("raw supplemental deficit 0.165-0.187, all p=0.0001",
               "0.165-0.187", f"{min(dd)}-{max(dd)}",
               all(0.165 <= round(d, 3) <= 0.187 for d in dd) and all(c["p"] == 0.0001 for c in dual)))
aa = json.load(open(os.path.join(here, "results", "views", "author_analysis.json")))
checks.append(("matched-length subset: all paired tests null (p 0.30-0.93)",
               ">=0.30", min(m["p"] for m in aa["matched"]),
               all(m["p"] >= 0.30 for m in aa["matched"])))
it = json.load(open(os.path.join(here, "results", "views", "interaction_test.json")))
checks.append(("interaction D in 0.014-0.020, CI excludes zero, all seeds",
               "0.014-0.020", f'{min(x["D"] for x in it["interaction"])}-{max(x["D"] for x in it["interaction"])}',
               all(x["excludes_zero"] and 0.014 <= x["D"] <= 0.020 for x in it["interaction"])))
rev = it["reversal"]
narr_sig = sum(1 for r in rev if r["view"] == "narr" and r["tfidf_minus_bilstm"] > 0 and r["p"] <= 0.0012)
syn_sig = sum(1 for r in rev if r["view"] == "syn" and r["tfidf_minus_bilstm"] < 0 and r["p"] <= 0.0002)
checks.append(("ranking reversal: sig 2/3 each direction", "2+2", f"{narr_sig}+{syn_sig}",
               narr_sig >= 2 and syn_sig >= 2))
ens = it["ensemble_holm"]
checks.append(("ensemble beats better single view, Holm <= 0.0104, all 6",
               "<=0.0104", max(e["p_holm"] for e in ens),
               all(e["delta"] > 0 and e["p_holm"] <= 0.0104 for e in ens)))

# --- NHTSA purpose hierarchy (v2 = duplicate-confined; 3m = class-vocab mask) ---
n2 = json.load(open(os.path.join(here, "results", "nhtsa", "nhtsa2_contrasts.json")))
sc = [c["delta"] for c in n2 if "summary vs conseq" in c["contrast"]]
sr = [c["delta"] for c in n2 if "summary vs remedy" in c["contrast"]]
checks.append(("NHTSA summary>consequence 0.094-0.118", "0.094-0.118", f"{min(sc)}-{max(sc)}",
               all(0.094 <= round(d, 3) <= 0.118 for d in sc)))
checks.append(("NHTSA summary>remedy 0.129-0.160", "0.129-0.160", f"{min(sr)}-{max(sr)}",
               all(0.129 <= round(d, 3) <= 0.160 for d in sr)))
checks.append(("NHTSA 11/12 contrasts Holm-significant", 11,
               sum(1 for c in n2 if c["p_holm"] <= 0.0024),
               sum(1 for c in n2 if c["p_holm"] <= 0.0024) == 11))
n3 = json.load(open(os.path.join(here, "results", "nhtsa", "nhtsa3m_contrasts.json")))
msr = [c["delta"] for c in n3 if "summary vs remedy" in c["contrast"]]
checks.append(("masked hierarchy: 12/12 Holm-significant, summary>remedy 0.112-0.166",
               "12; 0.112-0.166", f'{sum(1 for c in n3 if c["p_holm"] <= 0.0024)}; {min(msr)}-{max(msr)}',
               all(c["p_holm"] <= 0.0024 for c in n3) and all(0.112 <= round(d, 3) <= 0.166 for d in msr)))

bad = 0
for label, p, s, ok in checks:
    sv = round(s, 4) if isinstance(s, float) else s
    print(f"{'OK      ' if ok else 'MISMATCH'} {label}: paper {p}  shipped {sv}")
    bad += 0 if ok else 1
print(f"\n{len(checks)-bad}/{len(checks)} checks pass")
raise SystemExit(1 if bad else 0)
