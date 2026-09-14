# Analyses declared before their results were seen

Analyses added after protocol_maintnet.md, each specified before its own
results existed. Entries were written in our working folder before each
run and copied here afterwards; the git timestamps of this file are later
than the runs. The manuscript labels these "prospective extension", or
"post hoc" where stated below.

## Shared-representation controls (written 2026-08-24, post hoc)

Control 1: the interaction D with the mean-pooling model retrained on both
ASRS records over GloVe-200 (six runs), against the existing BiLSTM
predictions over GloVe-200.
Result: D = 0.0104 [0.0042, 0.0166], 0.0088 [0.0024, 0.0150], 0.0002
[-0.0060, 0.0064]. Positive in all runs; the interval excludes zero in two.
From the same runs: under shared GloVe-200 the mean-pooling model has a
synopsis advantage of +0.0077 (p = 0.0064), +0.0050 (p = 0.0794), +0.0151
(p = 0.0002). The statement that mean-pooled embeddings show no synopsis
advantage holds for record-trained embeddings only.

Control 2: the NHTSA field hierarchy with one word2vec trained on the
training text of all three fields and frozen for all three classifiers
(nine runs).
Result: summary 0.7358, 0.7838, 0.7913; consequence 0.6953, 0.6812, 0.6813;
remedy 0.7074, 0.7095, 0.6780. All six summary contrasts Holm-significant
(+0.0283 to +0.1133, weakest adjusted p 0.0464). Remedy minus consequence
+0.0121 (n.s.), +0.0283 (adjusted p 0.0486), -0.0032 (n.s.).

## RoBERTa contrasts (written 2026-08-25, before the ASRS fine-tunes finished)

Four contrasts, record-level paired inference, Holm within this family:

1. ASRS synopsis against narrative under RoBERTa, per run.
2. D_R = [S(syn, RoBERTa) - S(syn, BiLSTM)] - [S(narr, RoBERTa) - S(narr,
   BiLSTM)], record bootstrap; TF-IDF as secondary baseline.
3. NHTSA: G_v = S(v, RoBERTa) - S(v, BiLSTM) per field; record bootstrap of
   G_remedy - G_summary.
4. NHTSA summary field: TF-IDF against RoBERTa, paired test.

No further RoBERTa statistics. No 512-token run unless the ASRS contrast
were small enough for truncation to change its sign. All outcomes to be
reported as found.

## Negative-draw sensitivity (written 2026-08-26)

Five independent draws of the negative class (seeds 101 to 105) from the
same non-Aircraft pool, same positives, same split policy. Per draw: word
TF-IDF and a BiLSTM over word2vec trained on that draw, narrative and
synopsis, one training each. Statistic: synopsis minus narrative macro-F1
per model per draw. Criterion: direction and approximate size persist across
draws.
Result: BiLSTM +0.0041, +0.0064, +0.0089, +0.0090, +0.0510 (the last from a
narrative training that converged at 0.834); TF-IDF 0.000 to +0.0045.

## Two completions (written 2026-08-27)

1. Trainings 2 and 3 for the BiLSTM of each negative draw (torch seeds 101
and 102), each draw's word2vec shared across its trainings. Statistic: the
per-draw mean of the three synopsis minus narrative differences. Criterion:
positive in every draw.
Result: trainings 1, 2, 3 per draw: +0.051, +0.008, +0.007; +0.009, +0.048,
+0.004; +0.004, +0.007, +0.018; +0.006, +0.006, +0.012; +0.009, +0.042,
+0.013. Per-draw means +0.022, +0.020, +0.010, +0.008, +0.021. Criterion
met.

2. NHTSA temporal split: train on campaign years 2000 to 2021, test on 2022
to 2026 (3681 campaigns, 22.1%, the boundary closest to the 20% policy,
chosen before any model was run); duplicate groups spanning the boundary
lose their test-side members. Per field: TF-IDF and a BiLSTM over field
word2vec fitted on the temporal training text, three trainings. Tests:
summary minus consequence and summary minus remedy, Holm within the
six-contrast family. Only the field ordering is interpreted; absolute
scores are not comparable with the random split.
Result: 16163 campaigns after boundary-duplicate removal, 3218 evaluated.
All six contrasts significant (adjusted p 0.0006 each): summary minus
consequence +0.047, +0.082, +0.074; summary minus remedy +0.117, +0.109,
+0.154. TF-IDF, outside the family: +0.092 and +0.162, p 0.0001. The first
execution saved only aggregate scores and was rerun with predictions kept;
the first gave +0.109, +0.087, +0.113 and +0.164, +0.127, +0.174. The
tabulated run is the second. Both result files are in results/nhtsa/.

## NHTSA near-duplicate grouped split (written 2026-09-13)

Grouping: union-find over campaigns, starting from the exact-key groups of
the reference split, then joining any two campaigns whose token sets in
the same field have Jaccard similarity of 0.80 or more. Groups are shuffled
with the same seed (20260802) and filled into the test side until 20% of
campaigns. Everything else as in nhtsa_task.py. RoBERTa not rerun.
Criterion: all six BiLSTM summary contrasts positive and Holm-significant
at 0.05, and both TF-IDF summary contrasts positive. The consequence-remedy
ordering is descriptive only. Either way, the group counts, the number of
campaigns in multi-member groups and the train and test sizes are printed.
Result: criterion met. 8526 groups against 11570 under exact keys; 9450
campaigns in multi-member groups; largest group 4418. Summary minus
consequence +0.072 (TF-IDF), +0.077, +0.087, +0.096 (BiLSTM); summary minus
remedy +0.138, +0.144, +0.149, +0.147. tabulate_neardup.py prints the
tables.
