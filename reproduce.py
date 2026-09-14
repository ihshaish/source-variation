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

# --- shared-representation controls (post hoc, review round) ---
cg = json.load(open(os.path.join(here, "results", "views", "control_glove_D.json")))
gi = cg["interaction_glove"]
checks.append(("shared-GloVe D positive all runs, CI excludes zero in 2/3",
               "2/3", f'{sum(1 for x in gi if x["excludes_zero"])}/3',
               all(x["D"] >= 0 for x in gi) and sum(1 for x in gi if x["excludes_zero"]) == 2))
cn = json.load(open(os.path.join(here, "results", "nhtsa", "control_nhtsa_shared.json")))
summ = [c for c in cn["contrasts"] if c["contrast"].split(": ")[1].startswith("summary")]
rc = [c for c in cn["contrasts"] if "remedy vs conseq" in c["contrast"]]
checks.append(("shared-w2v NHTSA: all 6 summary contrasts Holm-significant",
               6, sum(1 for c in summ if c["p_holm"] <= 0.05),
               len(summ) == 6 and all(c["delta"] > 0 and c["p_holm"] <= 0.05 for c in summ)))
checks.append(("shared-w2v NHTSA: remedy-consequence gap closes (|d| <= 0.03)",
               "<=0.03", max(abs(c["delta"]) for c in rc),
               all(abs(c["delta"]) <= 0.03 for c in rc)))

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


# --- GE headline numbers, from the metrics exported out of the secure environment (results/ge/) ---
ge = [json.loads(l) for l in open(os.path.join(here, "results", "ge", "ge_results.jsonl"))]
def gemean(field, emb="avi2vec", model="bilstm"):
    v = [r["test_macro_f1"] for r in ge if r["key"].startswith(f"final_random_{field}_{emb}_{model}_s") and r["key"].count("_") == 5]
    assert len(v) == 3, (field, emb, model, len(v))
    return sum(v) / 3
cust, tech, rep = gemean("customer"), gemean("technician"), gemean("repair")
claim("GE customer field (Avi2Vec BiLSTM, three-training mean)", 0.327, cust)
claim("GE technician field", 0.783, tech)
claim("GE repair-action field", 0.910, rep)
claim("GE customer -> technician difference", 0.456, tech - cust)
claim("GE customer -> repair action difference", 0.583, rep - cust)
claim("GE Avi2Vec vs GloVe-200 on repair action", 0.049, rep - gemean("repair", "glove200"))
claim("GE sequence vs mean pooling (GloVe-200, repair action)", 0.092, gemean("repair", "glove200") - gemean("repair", "glove200", "meanmlp"))
lk = json.load(open(os.path.join(here, "results", "ge", "leakage_report.json")))
claim("GE keyword rule over curated outcome terms, repair action", 0.292, lk["keyword_baseline_macro_f1"]["repair"])

# --- Table 2 seed-averaged differences with joint record-resampling intervals ---
sa = json.load(open(os.path.join(here, "results", "views", "seedavg_table4.json")))
claim("Table 2 ASRS synopsis - narrative, BiLSTM mean", 0.015, sa["syn_narr_bilstm"]["mean"])
claim("Table 2 ASRS synopsis - narrative, RoBERTa mean", 0.015, sa["syn_narr_roberta"]["mean"])
claim("Table 2 cross-record transfer T, mean", 0.179, sa["dual_raw"]["mean"])
claim("Table 2 comparable-length subset, mean", 0.010, sa["dual_matched"]["mean"])

# --- five redrawn negative samples (Supplementary Table S14) ---
sd = json.load(open(os.path.join(here, "results", "views", "sensitivity_draws.json")))
se = json.load(open(os.path.join(here, "results", "views", "sensitivity_extra.json")))
means = []
for d in range(1, 6):
    v = [r["bilstm_delta"] for r in sd if r["draw"] == d] + [r["bilstm_delta"] for r in se if r["draw"] == d]
    assert len(v) == 3, d
    means.append(sum(v) / 3)
checks.append(("redraws: BiLSTM synopsis advantage positive in every draw, means in 0.008-0.022",
               "0.008-0.022", f"{round(min(means),3)}-{round(max(means),3)}",
               all(m > 0 for m in means) and 0.008 <= round(min(means), 3) and round(max(means), 3) <= 0.022))

# --- NHTSA temporal split (Supplementary Table S17) ---
tt = json.load(open(os.path.join(here, "results", "nhtsa", "nhtsa_temporal_tests.json")))
ts = tt["scores"]
for f, val in [("summary", 0.718), ("conseq", 0.651), ("remedy", 0.592)]:
    claim(f"temporal split, BiLSTM {f} mean", val, sum(ts[f"bilstm_{f}_s{s}"] for s in range(3)) / 3)
fam = tt["family_holm"]
checks.append(("temporal: all six declared BiLSTM summary contrasts positive and Holm-significant",
               6, sum(1 for c in fam if c["delta"] > 0 and c["p_holm"] < 0.05),
               len(fam) == 6 and all(c["delta"] > 0 and c["p_holm"] < 0.05 for c in fam)))

# --- NHTSA near-duplicate grouped split (Supplementary Table S18) ---
nd = {r["key"]: r["f1"] for r in json.load(open(os.path.join(here, "results", "nhtsa", "neardup", "nhtsaND_results.json")))}
for f, val in [("summary", 0.731), ("conseq", 0.644), ("remedy", 0.584)]:
    claim(f"near-duplicate split, BiLSTM {f} mean", val, sum(nd[f"nhtsa_{f}_bilstm_s{s}"] for s in range(3)) / 3)
nc = json.load(open(os.path.join(here, "results", "nhtsa", "neardup", "nhtsaND_contrasts.json")))
bil = [c for c in nc if c["contrast"].startswith("bilstm") and "summary vs" in c["contrast"]]
checks.append(("near-duplicate: all six BiLSTM summary contrasts positive and Holm-significant (registered criterion)",
               6, sum(1 for c in bil if c["delta"] > 0 and c["p_holm"] < 0.05),
               len(bil) == 6 and all(c["delta"] > 0 and c["p_holm"] < 0.05 for c in bil)))
ng = json.load(open(os.path.join(here, "results", "nhtsa", "neardup", "nhtsaND_grouping.json")))
claim("near-duplicate grouping: campaigns in multi-member groups", 9450, ng["campaigns_in_multi_groups"], 0)
claim("near-duplicate grouping: largest transitive group", 4418, ng["largest_group"], 0)

# --- RoBERTa suite (Supplementary Tables S19-S20) ---
rb = json.load(open(os.path.join(here, "results", "views", "control_roberta.json")))
rres = {r["key"]: r["f1"] for r in rb["results"]}
claim("RoBERTa ASRS synopsis - narrative, mean over seeds", 0.015,
      sum(rres[f"roberta_asrs_syn_s{s}"] - rres[f"roberta_asrs_narr_s{s}"] for s in range(3)) / 3, 0.001)
pr = json.load(open(os.path.join(here, "results", "views", "control_roberta_prereg.json")))
dr_tf = [x for x in pr["DR"] if x["baseline"] == "tfidf"]
checks.append(("RoBERTa D against TF-IDF: every interval excludes zero", 3,
               sum(1 for x in dr_tf if x["ci"][0] > 0), all(x["ci"][0] > 0 for x in dr_tf)))

bad = 0
for label, p, s, ok in checks:
    sv = round(s, 4) if isinstance(s, float) else s
    print(f"{'OK      ' if ok else 'MISMATCH'} {label}: paper {p}  shipped {sv}")
    bad += 0 if ok else 1
print(f"\n{len(checks)-bad}/{len(checks)} checks pass")
raise SystemExit(1 if bad else 0)
