# Register addendum v5.1 — shared-representation controls (2026-08-24)

Two post-hoc controls sanctioned by Hisham after the pre-submission review,
run on the frozen tasks and splits with the frozen training protocol. Both
interpretations were pre-committed before the runs (see control script
docstrings, committed with the code).

## Control 1 — interaction D under shared GloVe-200 (ASRS)

Mean-pool MLP retrained on both views with GloVe-200 (6 runs); D recomputed
against the existing BiLSTM/GloVe-200 predictions.

D = 0.0104 [0.0042, 0.0166], 0.0088 [0.0024, 0.0150], 0.0002 [-0.0060, 0.0064].

Outcome: positive in all runs, CI excludes zero in 2 of 3. CLAIM 6 STANDS,
now stated with the control: "most of the interaction persists without
view-trained representations, at reduced size and less uniformly."

Secondary finding (unplanned, from the same runs, paired-tested on stored
predictions): under shared GloVe-200, MEAN POOLING shows a synopsis
advantage (+0.0077 p=0.0064, +0.0050 p=0.0794, +0.0151 p=0.0002).
CLAIM 1 SCOPE CORRECTED: "mean-pooled embeddings show no corresponding
advantage" holds for VIEW-TRAINED embeddings only. New statement: the
advantage is absent under lexical baselines, representation-dependent under
mean pooling (absent with view-trained embeddings, present at reduced size
under shared GloVe-200), and present in every sequence-model run.
Abstract wording now "under sequence models, not lexical baselines"
(replacing "under sequence models only").

## Control 2 — NHTSA hierarchy under one shared word2vec

One word2vec trained on the concatenation of the three fields' training
text, frozen for all three field classifiers (9 runs).

summary 0.7358/0.7838/0.7913; conseq 0.6953/0.6812/0.6813;
remedy 0.7074/0.7095/0.6780.
Summary contrasts: all 6 Holm-significant (+0.0283 to +0.1133; weakest
p_holm 0.0464). Remedy-vs-conseq: +0.0121 ns / +0.0283 p_holm 0.0486 /
-0.0032 ns — gap CLOSES, sign mixed.

Outcome: CLAIM 3 SCOPED, not weakened at the top: the summary's primacy is
representation-independent (and survives class-vocab masking, from the main
programme); the consequence-over-remedy ordering under the BiLSTM is partly
a property of the field-trained representations (remedy gains most from the
shared vocabulary). Manuscript states: "The top of the hierarchy is
representation-independent; the ordering of the two lower fields is not."
The frozen claim's hierarchy sentence remains accurate for the as-deployed
per-field pipelines (TF-IDF and field-trained BiLSTM, 11/12 Holm; masked
12/12).

## Provenance correction (same session, review-driven)

NHTSA target re-described per NHTSA's own documentation: the component
classification is determined by NHTSA's analysis of the manufacturer's Part
573 report (Recall Completion Rates Report to Congress), not "the same
filing"; all "author and moment fixed" phrasing removed (restores the
register's own ban on "same author/one moment"). Claim 3's propositions
(numbers, model-independence, mask survival, interpretation) unchanged.

Everything else in CLAIMS_FREEZE_v5 is untouched.

## Pre-declared RoBERTa analysis plan (2026-08-25, BEFORE the ASRS runs completed)

Declared while the narrative fine-tunes are still training; the NHTSA and
synopsis halves are complete but no ASRS view contrast has been computed.

Contrasts (record-level paired inference, Holm within this family only):
1. ASRS syn vs narr under RoBERTa, per run (the script already computes this).
2. D_R = [S(syn,RoBERTa)-S(syn,BiLSTM_view)] - [S(narr,RoBERTa)-S(narr,BiLSTM_view)],
   record bootstrap; secondary baseline TF-IDF.
3. NHTSA gain comparison: G_v = S(v,RoBERTa)-S(v,BiLSTM_field); compare
   G_remedy vs G_summary (record bootstrap of the difference).
4. NHTSA summary: TF-IDF vs RoBERTa paired test (is the lexical model's edge
   on the summary field real or parity?).
No further RoBERTa statistics. No 512-token run unless the ASRS view
contrast is small enough that truncation could plausibly change its sign.

Interpretation matrix, fixed in advance:
- syn > narr significant: the analyst-view contrast extends to a fine-tuned
  contemporary encoder (claim 1 scope widens; claim 2 gains D_R evidence).
- syn ~ narr null: the contrast is model-regime dependent; claim 1's ASRS
  component is scoped to the earlier families, and claim 2 (model
  conclusions change with record/model regime) is STRENGTHENED.
- narr > syn reversed: strongest claim-2 evidence; reported as such.
Every outcome is reportable; none is adverse to the reframed paper.

## Pre-declared undersample sensitivity (2026-08-26, BEFORE any results)

Sanctioned by Hisham after two external reviews independently identified the fixed
negative-class undersample as the most material remaining uncertainty. Design: five
independent negative-class draws (seeds 101-105) from the same non-Aircraft pool,
same positives, same 80/20 split policy; per draw, word TF-IDF and BiLSTM over
draw-trained word2vec (frozen protocol constants), narrative and synopsis, one
training each; statistic = synopsis minus narrative macro-F1 per model per draw on
that draw's held-out set. Interpretation fixed in advance: contrast direction and
approximate size persist across draws -> the analyst-view effect does not depend on
the original negative sample; sign flips or collapse -> reported as draw-dependence
and claim 1 scoped accordingly. No further statistics from these runs.

OUTCOME (2026-08-26, all five draws complete): synopsis minus narrative positive
under the BiLSTM in every draw (+0.0041/+0.0064/+0.0089/+0.0090/+0.0510; the
largest driven by one weakly converged narrative training, narr 0.834), and
0.000 to +0.0045 under TF-IDF, as on the primary task. Per the pre-committed
interpretation: the analyst-record effect does not depend on the original
negative sample. Integrated as Supplementary Table S10 + one sentence in 5.2.

## Pre-declared completions (2026-08-27, BEFORE any results; review-4 sanctions)

1. Negative-draw completion: trainings 2 and 3 (torch seeds 101, 102) added for the
BiLSTM configurations of each of the five negative draws, identical protocol and
draw-trained word2vec; statistic unchanged (synopsis minus narrative per draw and
training). Interpretation as originally declared; the added trainings separate
draw variation from training variation.

2. NHTSA temporal robustness: same 16-class task recipe (any-field exact-duplicate
grouping, classes >=300); split by campaign year encoded in the campaign number —
train 2000-2021, test 2022-2026 (3681 campaigns, 22.1%, the boundary closest to the
20% policy, chosen from the year distribution BEFORE any model was run); duplicate
groups spanning the boundary have their test-side members removed. Per field:
word TF-IDF and BiLSTM over field-trained word2vec fitted on the temporal training
text (frozen constants), three trainings. Statistics: summary-consequence and
summary-remedy contrasts on the temporal test set. Interpretation fixed in advance:
hierarchy direction persists out of time -> temporal robustness of claim 3;
weakened or reversed -> reported as temporal sensitivity, scoping claim 3.
No further statistics from either completion.

Clarifications (2026-08-27, minutes after launch; temporal run mid-training with no
contrast yet computed, no extra-draw result yet existing): (a) negative-draw completion
statistic = per-draw MEAN of the three trainings' synopsis-minus-narrative deltas, with
the direction criterion (mean positive in every draw) as declared; each draw's word2vec
is retrained deterministically (fixed seed 20260802), so trainings within a draw share
the identical embedding. (b) NHTSA temporal contrasts are Holm-corrected within the
six-contrast family (2 contrasts x 3 trainings); absolute temporal-test macro-F1 is NOT
comparable to the random-split values (class and language drift 2022-2026) and only the
field ordering within the temporal test set carries the interpretation.

## Scoping note v5.2 (2026-08-27; presentational, logged per the v5.1 pattern)
(1) NHTSA masking headline now reads "not explained by literal class-label vocabulary"
(evidence chain unchanged: 12/12 Holm-significant masked; the literal scoping the freeze
itself carries in "extends beyond literal component-name echo" is made explicit at the
headline). (2) "The top of the hierarchy is representation-independent; the lower
ordering is not" now reads "The summary's advantage persists across the representations
tested; the ordering of the two lower fields does not." Substance unchanged in both.


## Outcome: NHTSA temporal completion (logged 2026-08-26)
Executed as declared. Task: 16163 campaigns after the declared boundary-duplicate
removal (3681 test campaigns by year; 463 test-side members of boundary-spanning
duplicate groups removed; 3218 evaluated). RESULT: hierarchy direction persists out of
time — the summary is strongest under both models in every training. Pre-declared
six-contrast family (2 summary contrasts x 3 BiLSTM trainings), approximate
randomisation 10000 permutations, Holm within the family: all six significant
(p_Holm=0.0006 each). BiLSTM summary-consequence +0.047/+0.082/+0.074;
summary-remedy +0.117/+0.109/+0.154. TF-IDF (descriptive, outside the family):
+0.092 and +0.162, unadjusted p=0.0001. Interpretation per the declaration: temporal
robustness of claim 3. Reported in main Results (Section 6.2 NHTSA paragraph) and
Supplementary Table S16; status row added to the S7 register.
EXECUTION NOTE: the first execution (nhtsa_temporal.py) saved only aggregate F1s, so
the declared Holm tests could not be computed from it; the pipeline was re-executed
identically with per-record predictions retained (nhtsa_temporal_preds.py). TF-IDF
reproduces exactly; BiLSTM trainings differ between executions (training is not
bit-reproducible on the MPS hardware), first execution summary-consequence
+0.109/+0.087/+0.113 and summary-remedy +0.164/+0.127/+0.174 — same direction on
every contrast in both executions. The tabulated run is the one with retained
predictions, on which the pre-specified tests were computed. Both result files are
archived (nhtsa_temporal.json, nhtsa_temporal_tests.json, nhtsa_temporal_preds.npz).


## Outcome: negative-draw completion (logged 2026-08-26)
Executed as declared (clarification a): BiLSTM trainings 2 and 3 (torch seeds 101, 102)
per draw, identical protocol, each draw's word2vec shared across its trainings.
Per-training deltas (trainings 1/2/3): draw 1 +0.051/+0.008/+0.007; draw 2
+0.009/+0.048/+0.004; draw 3 +0.004/+0.007/+0.018; draw 4 +0.006/+0.006/+0.012;
draw 5 +0.009/+0.042/+0.013. Per-draw MEANS: +0.022, +0.020, +0.010, +0.008, +0.021 —
positive in every draw, so the pre-declared direction criterion is met and claim
robustness to the negative sample stands on three trainings per draw. Weakly converged
narrative trainings recur across draws and torch seeds (draw 2 training 2 narrative
0.835; draw 5 training 2 narrative 0.840; cf. the original draw 1 training 1), which
supports treating them as ordinary training variability rather than a property of one
draw; the single-training caveat is accordingly removed from the main text and the full
per-training spread retained in Supplementary Table S13. Archived in
sensitivity_draws.json + sensitivity_extra.json.


## Scoping note v5.3 (2026-08-26; review-5 execution, logged BEFORE editing)
All claim substance unchanged. Presentational changes approved by Hisham:
(1) Abstract's secondary claim simplified to "Secondary analyses showed that some
conclusions about modelling approaches were also record-dependent"; the frozen
reversal claim stays verbatim in Results/Discussion.
(2) Sec 6.3 categorical sentence rescoped: "In these comparisons, the record
dependence is concentrated between lexical and learned sequence models; RoBERTa and
the BiLSTM do not show a consistent record-dependent difference" (substance = the
four pre-specified RoBERTa contrasts, unchanged).
(3) The fixed-model primary→supplemental headline number (0.165–0.187 / 0.179) is
now labelled a cross-record TRANSFER quantity, T(r_a→r_b;m); the label attaches to
the number only — Claim 2's completeness-and-curation interpretation is unchanged
(training on the supplemental record does not remove the deficit).
(4) S(r,m) clarified as the score of a fixed modelling PROCEDURE fitted per record
(end-to-end contrast, procedure refitted); notation clarification only.
(5) Table 2 regrouped into Panel A (matched, separately fitted; one declared
configuration per row) and Panel B (transfer + secondary). NEW numbers thereby
printed: NHTSA BiLSTM-only ranges — summary−consequence +0.113 to +0.118,
summary−remedy +0.129 to +0.160 (from the frozen per-run table); the frozen
model-spanning ranges 0.094–0.118 and 0.129–0.160 move verbatim into 6.1 prose.
Claim-2's six-training/two-embedding span 0.165–0.187 keeps its definition (stated
in Panel B caption).
(6) S7 status vocabulary: "preplanned (original study)" / "declared at revision,
before the corresponding results" / "post hoc (exploratory)". "Corresponding"
mandatory. Register-internal terminology unchanged.
(7) Rejected reviewer edit, logged: no "solely" in the NHTSA masking headline — the
sentence is Claim-3 freeze substance scoped to the ordering, which survives complete
literal-vocabulary masking (12/12 Holm); "solely" would weaken a frozen sentence.
(8) New interpretive content (no claims): information-state distinction paragraph;
agentic-AI implication paragraph (+2 citations: Crespo-Márquez & Gómez Fernández,
C&IE 2026; Farahani, Khan & Wuest, JMS 2026); industrial resource-allocation framing.
Freeze-protected significance statements exempt from the language pass (verbatim):
"11 of 12 paired contrasts Holm-significant"; masked "twelve of twelve"; temporal
"all six pre-specified BiLSTM contrasts Holm-significant"; Claim-4 "p_holm <=
0.0104" family; Claim-6 interval-exclusion statements.

## v5.3 execution note (2026-08-26)
All v5.3 items executed as logged. Additional presentational corrections during QA:
Claim-6 control sentence restored to the v5.1 qualified form ("at reduced size, though
less uniformly"); masked-hierarchy significance ("all twelve masked contrasts
Holm-significant") and fusion significance ("all p_holm <= 0.0104") reinstated in main
text per the exempt list; Table 2 Panel-B caption states per-row estimators (transfer
= word2vec mean; comparable-length = six-training mean; both ranges span both
embeddings); per-run G_summary/G_remedy added to supplement Table S18 from the
archived control_roberta_prereg.json; the per-field gain interval claim rescoped to
the archived remedy-summary difference intervals. No claim substance changed.


## Scoping note v5.4 (2026-08-26; review-6 execution, logged BEFORE editing)
ERRATUM (numeric, with provenance): Supplementary Table S10's comparable-length
(matched-subset, n=571, ratio>=0.7) block did not match the archived predictions.
Direct recomputation from dualpreds_{w2vview,glove200}_s{0,1,2}.npz (deterministic
arithmetic on stored predictions): per-run deltas w2v +0.0118/+0.0168/+0.0008 (mean
0.0098 — the registered "+0.010 [−0.013, 0.033]" is exactly this w2v estimator, so
the registered mean and CI are CONFIRMED); GloVe-200 +0.0166/+0.0184/+0.0051.
Corrections: S10 matched rows replaced; their paired p-values rerun from stored
predictions; main Table 2 matched-row range "−0.001 to +0.017" → "+0.001 to +0.017"
(w2v, one population); the six-run span quoted in prose → "+0.001 to +0.018"
(agrees with Table S12, which was correct). Claim substance unchanged (comparable-
length null) and marginally strengthened (all six deltas non-negative).
Presentational (approved): Table 2 rows single-population (transfer row w2v range
+0.170 to +0.187); Fig 4 caption states the two smallest entries are family upper
bounds (same fix in the supplement's expanded figure); Tables 3 and 4 leave the main
text with all values discoverable in the supplement; key-panel per-class values,
class percentages, Jaccard 0.34 and audit co-investigator detail move to the
supplement; identifiability bound removed (descriptive maths, no claim rests on it);
6.2/6.3/6.4 compressions with the exempt-list statements and the Claim-1
mean-pooling scope sentence as named survivors; first-person definition voice (H-1);
keyword "data-centric AI" → "industrial data quality"; agentic paragraph shortened
retaining "information boundary". New printed numbers: 0.362–0.891 (lexical-gradient
sentence), Table 2 w2v ranges, corrected S10 values.

## v5.4 execution note (2026-08-26)
All v5.4 items executed, incl. the S10→S11 matched-block erratum (values + rerun
p-values from archived predictions; disclosed in the table caption). QA additions
during the gates: main 5.3's RoBERTa declaration wording aligned to the git record
("before its ASRS fine-tunes completed", matching Table S19's caption); the
comparable-length sentence split so the three-run w2v estimator cannot be read as a
six-run mean; Table S3 gained the GE analysis-family row (permutations 5000);
"views"/"View" terminology retired from the supplement; Fig 4/Fig S2 captions scope
the upper-bound families to the three record fields. Claim substance unchanged
throughout; exempt statements verified verbatim by the orchestrator, which also
recomputed ~30 printed numbers against the per-run tables (all reconcile) and
cleared the shows/does-not-show boundary list.


## Scoping note v5.5 (2026-08-26; review-7 final editorial pass, logged BEFORE editing)
All presentational; no claim substance changes. (1) Analysis-status vocabulary in
the PUBLISHED documents becomes preplanned / prospective extension (specified before
that analysis was run) / post hoc (exploratory); "at revision" removed everywhere as
project history, not evidential status. THIS REGISTER keeps its own dated
terminology unchanged — it remains the internal chronology of record. (2) Title
changes to "The record is part of the task: matched-record evaluation of text
classifiers across maintenance, safety, and recall reporting" (flagged to Hisham for
veto); keyword set replaced accordingly. (3) Table 2 GE rows gain printed mean
contrasts +0.456 and +0.127 (differences of the frozen field means 0.327/0.783/
0.910; new printed rounding 0.127 logged) with a table note stating GE bootstrap
endpoints were not exported from the secure environment. (4) Figure 1 and its
caption move to modelling-procedure terminology (matches the v5.3 clarification).
(5) The 33-item language list + question-title renames (6.1, 6.3, 6.2) + hyphen
reduction: rewordings only; the R-14 exempt statements and frozen sentences remain
verbatim except where a listed item names them (none does). (6) Supplement drafting-
history sentences removed (initial temporal execution note; "correcting an earlier
tabulation"; RoBERTa no-further-statistics line); the register and repository remain
the record of those facts. (7) Reporting-table schema gains producer and record
purpose columns; caption "Recommended reporting information for record-based text
classification".


## Prospective extension v5.6 (2026-09-13; NHTSA near-duplicate grouped split; logged BEFORE the run)
Purpose. The registered NHTSA split (leg 2) confines campaigns sharing an IDENTICAL
text in any of the three fields (key: lowercased text, first 400 characters) to one
side of the 80/20 campaign split. Manufacturer boilerplate that differs by a few
tokens is not caught by that key, so template overlap between partitions remains
possible. This extension re-runs the NHTSA leg with near-duplicates grouped as well.

Grouping rule (fixed now). Union-find over campaigns, starting from the leg-2
exact-key unions, then additionally uniting any two campaigns whose token sets in
the SAME field (Summary, Consequence or Remedy; tokeniser and lowercasing identical
to the leg, empty fields ignored) have Jaccard similarity >= 0.80. Groups are then
shuffled with the leg-2 seed (20260802) and filled into the test side until 20% of
campaigns is reached, exactly as in leg 2. Everything else is held fixed: the 16
classes (support >= 300), field word2vec per field trained on the training side,
TF-IDF (1-2 grams, min_df 3, sublinear) with logistic regression, BiLSTM (frozen
200-d embeddings, hidden 64, cap 96, dropout 0.3, Adam 1e-3, 15 epochs, patience 2,
5% validation, seeds 700-702), macro-F1 on the held-out side, paired prediction-swap
randomisation tests (5000 permutations) for the three field pairs per training, Holm
within the twelve-contrast family as in leg 2. RoBERTa is not re-run (comparator only).

Pre-committed criterion ("hierarchy persists"). Claim 3 as printed is summary-first:
the defect summary exceeds both the consequence and the remedy field under both
models in every training. The criterion is therefore: all six BiLSTM summary
contrasts (summary - consequence, summary - remedy, three trainings) positive and
Holm-significant at 0.05, and both TF-IDF summary contrasts positive. The
consequence-remedy ordering is reported descriptively; the frozen text already
states that the ordering of the two lower fields does not persist under every
sensitivity condition, so it is not part of the criterion.

Pre-committed interpretation. Criterion met: the summary-first hierarchy is robust
to near-duplicate boilerplate; reported as one sentence in main 6.2 after the
temporal sentence, one supplement table (S4.3, after Table S17), one S7 row
"Near-duplicate grouped split (Jaccard >= 0.80, any field) - NHTSA - prospective
extension"; absolute scores are not compared with the random split. Criterion not
met (any summary contrast non-positive or non-significant): reported in the same
three places as a scoping sensitivity, the 6.2 sentence stating which contrasts
weakened, and the Discussion limitations gain one sentence that template overlap
contributes to the NHTSA hierarchy. Either way the group count, the number of
campaigns in multi-member groups and the change in train/test sizes relative to
leg 2 are printed so the reader sees how much the grouping moved.
Script: views_wip/nhtsa_neardup.py (leg-2 copy, split-grouping function replaced);
outputs under views_wip/neardup/ with prefix nhtsaND_.
