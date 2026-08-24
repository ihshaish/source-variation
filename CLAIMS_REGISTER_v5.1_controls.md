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
