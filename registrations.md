# Analyses declared before their results were seen

This file records the analyses that were added after the original plan
(protocol_maintnet.md) and specified before their own results existed. Each
entry gives the date it was written, what was fixed in advance and what came
out. The entries were written in our working folder before each run and
copied here afterwards, so the git timestamps of this file are later than
the runs. The manuscript labels these analyses "prospective extension", or
"post hoc" where the entry says so.

## Shared-representation controls (written 2026-08-24)

Two post hoc controls, run on the frozen tasks and splits with the frozen
training settings. Their interpretation was written down before the runs.

Control 1, the interaction D under one shared embedding (ASRS). The
mean-pooling model was retrained on both records with GloVe-200 (six runs)
and D recomputed against the existing BiLSTM predictions over GloVe-200.

Outcome: D = 0.0104 [0.0042, 0.0166], 0.0088 [0.0024, 0.0150] and 0.0002
[-0.0060, 0.0064]. Positive in all three runs, with the interval excluding
zero in two of them. The interaction persists without record-trained
representations, at reduced size and less uniformly. A secondary finding
from the same runs, paired-tested on the stored predictions: under shared
GloVe-200 the mean-pooling model shows a synopsis advantage of +0.0077
(p = 0.0064), +0.0050 (p = 0.0794) and +0.0151 (p = 0.0002). The earlier
statement that mean-pooled embeddings show no synopsis advantage therefore
holds for record-trained embeddings only, and the paper says so.

Control 2, the NHTSA field hierarchy under one shared word2vec. One word2vec
model trained on the training text of all three fields together and frozen
for all three field classifiers (nine runs).

Outcome: summary 0.7358, 0.7838, 0.7913; consequence 0.6953, 0.6812,
0.6813; remedy 0.7074, 0.7095, 0.6780. All six summary contrasts are
Holm-significant (+0.0283 to +0.1133, weakest adjusted p 0.0464). Remedy
against consequence: +0.0121 (not significant), +0.0283 (adjusted p
0.0486), -0.0032 (not significant); that gap closes and its sign is mixed.
The summary's advantage does not depend on the representation; the ordering
of the two lower fields does.

## RoBERTa contrasts (written 2026-08-25, before the ASRS fine-tunes finished)

Written while the narrative fine-tunes were still training; the NHTSA and
synopsis halves were complete and no ASRS record contrast had been computed.
Four contrasts, with record-level paired inference and Holm correction
within this family only:

1. ASRS synopsis against narrative under RoBERTa, per run.
2. D_R = [S(syn, RoBERTa) - S(syn, BiLSTM)] - [S(narr, RoBERTa) - S(narr,
   BiLSTM)] with a record bootstrap; TF-IDF as a secondary baseline.
3. NHTSA gain comparison: G_v = S(v, RoBERTa) - S(v, BiLSTM) per field, and
   the record bootstrap of G_remedy - G_summary.
4. NHTSA summary field: TF-IDF against RoBERTa, paired test.

No further RoBERTa statistics, and no 512-token run unless the ASRS contrast
turned out small enough for truncation to change its sign. Interpretation
fixed in advance: a significant synopsis advantage extends the record
contrast to a fine-tuned pretrained encoder; a null means the contrast
depends on the model family; a reversal is the strongest evidence that model
conclusions change with the record. All three outcomes were to be reported
as found.

## Negative-draw sensitivity (written 2026-08-26, before any result)

The ASRS task uses one fixed draw of the negative class. Five independent
draws (seeds 101 to 105) from the same non-Aircraft pool, with the same
positives and the same 80/20 split policy. Per draw: word TF-IDF and a
BiLSTM over word2vec trained on that draw, narrative and synopsis, one
training each. Statistic: synopsis minus narrative macro-F1 per model per
draw on that draw's held-out set. Interpretation fixed in advance: if the
direction and approximate size persist across draws, the record effect does
not depend on the original negative sample; sign flips or a collapse would
be reported as draw dependence. No further statistics from these runs.

Outcome: synopsis minus narrative positive under the BiLSTM in every draw
(+0.0041, +0.0064, +0.0089, +0.0090, +0.0510; the largest comes from one
weakly converged narrative training at 0.834) and 0.000 to +0.0045 under
TF-IDF, as on the primary task.

## Two completions (written 2026-08-27, before any result)

1. Negative draws, trainings 2 and 3. Torch seeds 101 and 102 added for the
BiLSTM of each of the five draws, same settings, each draw's word2vec shared
across its trainings (the word2vec is retrained with a fixed seed, so it is
identical within a draw). Statistic: the per-draw mean of the three
trainings' synopsis minus narrative differences; criterion: that mean
positive in every draw.

Outcome: per-training differences (trainings 1, 2, 3) of +0.051, +0.008,
+0.007 for draw 1; +0.009, +0.048, +0.004 for draw 2; +0.004, +0.007,
+0.018 for draw 3; +0.006, +0.006, +0.012 for draw 4; +0.009, +0.042,
+0.013 for draw 5. Per-draw means +0.022, +0.020, +0.010, +0.008, +0.021,
positive in every draw, so the criterion is met. Weakly converged narrative
trainings recur across draws and seeds (draw 2 training 2 at 0.835, draw 5
training 2 at 0.840), which supports treating them as ordinary training
variability.

2. NHTSA temporal split. The same 16-class task, split by the campaign year
encoded in the campaign number: train on 2000 to 2021, test on 2022 to 2026
(3681 campaigns, 22.1%, the year boundary closest to the 20% policy, chosen
from the year distribution before any model was run). Duplicate groups that
span the boundary lose their test-side members. Per field: word TF-IDF and
a BiLSTM over field word2vec fitted on the temporal training text, three
trainings. Statistics: summary minus consequence and summary minus remedy
on the temporal test set, Holm-corrected within the six-contrast family (two
contrasts by three trainings). Absolute scores on the temporal test set are
not comparable with the random split; only the field ordering carries the
interpretation. Interpretation fixed in advance: the hierarchy persisting
out of time supports the NHTSA claim; a weakened or reversed hierarchy would
be reported as temporal sensitivity.

Outcome: 16163 campaigns after the boundary-duplicate removal (3681 test
campaigns by year, 463 test-side members of boundary-spanning groups
removed, 3218 evaluated). The summary is strongest under both models in
every training. All six declared contrasts are significant (adjusted p
0.0006 each): BiLSTM summary minus consequence +0.047, +0.082, +0.074 and
summary minus remedy +0.117, +0.109, +0.154. TF-IDF, descriptive and
outside the family: +0.092 and +0.162, unadjusted p 0.0001. The first
execution of this split saved only aggregate scores, so it was run again
with predictions kept; the first execution gave +0.109, +0.087, +0.113 and
+0.164, +0.127, +0.174, the same direction on every contrast. The tabulated
run is the second one, on which the declared tests were computed, and both
result files are in results/nhtsa/.

## NHTSA near-duplicate grouped split (written 2026-09-13, before the run)

The registered NHTSA split confines campaigns that share an identical text
in any field (lower-cased, first 400 characters) to one side of the split.
Boilerplate that differs by a few tokens escapes that key. This extension
re-runs the NHTSA task with near-duplicates grouped as well.

Grouping rule: union-find over campaigns, starting from the exact-key
groups, then joining any two campaigns whose token sets in the same field
have Jaccard similarity of 0.80 or more (same tokeniser, empty fields
ignored). Groups are shuffled with the same seed (20260802) and filled into
the test side until 20% of campaigns is reached, as before. Everything else
stays fixed: the 16 classes, field word2vec trained on the training side,
TF-IDF with logistic regression, the BiLSTM settings, macro-F1 on the
held-out side, paired prediction-swap tests with 5000 permutations for the
three field pairs per training, Holm within the twelve-contrast family.
RoBERTa is not re-run.

Criterion: all six BiLSTM summary contrasts (summary minus consequence,
summary minus remedy, three trainings) positive and Holm-significant at
0.05, and both TF-IDF summary contrasts positive. The consequence-remedy
ordering is reported descriptively and is not part of the criterion.
Interpretation fixed in advance: criterion met, the summary-first hierarchy
holds under near-duplicate grouping and is reported in one sentence of the
Results, one supplement table and one status row; criterion not met, the
same three places report which contrasts weakened and the Discussion gains
a sentence on template overlap. Either way the group count, the number of
campaigns in multi-member groups and the change in train and test sizes are
printed.

Outcome: criterion met. 8526 groups against 11570 under exact keys, 9450
campaigns in multi-member groups, the largest group 4418 campaigns. Summary
minus consequence +0.072 (TF-IDF) and +0.077, +0.087, +0.096 (BiLSTM);
summary minus remedy +0.138 and +0.144, +0.149, +0.147; every BiLSTM
contrast Holm-significant. tabulate_neardup.py prints the full tables.
