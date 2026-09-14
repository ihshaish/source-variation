# source-variation

Code and results for "The record is part of the task: matched-record
evaluation of text classifiers across maintenance, safety and recall
reporting", submitted to Computers in Industry.

The study compares classifiers that read different records of the same
cases, with the cases, labels and split held fixed. Three systems: GE
Aerospace repair events (customer report, technician report, repair
action), NASA ASRS safety reports (reporter narrative, supplemental
narrative, analyst synopsis) and NHTSA recall campaigns (defect summary,
consequence, remedy).

Questions and mismatches: hisham.ihshaish@uwe.ac.uk, peter.mayhew@geaerospace.com.

## Checking the paper's numbers

```
python3 reproduce.py
```

Reads results/ and prints each headline number of the paper next to the
stored value, 54 checks. No training, no downloads.

## What runs here

- asrs_pipeline/ and records/ run from a fresh clone plus the ASRS export
  and the GloVe vectors, which you download yourself.
- nhtsa/ runs from the snapshot shipped in the repository.
- ge_package/ ran inside GE Aerospace. The records are proprietary and
  cannot be shared, and so are the Avi2Vec vectors trained on them. The
  package ships with a self-test on invented records. results/ge/ holds the
  exported metrics. Nothing under results/ contains record text.

## ASRS data

Export from https://asrs.arc.nasa.gov/search/database.html as CSV, year by
year (the interface caps results per query), into one directory. The task
uses reports through 2021; the corpus statistics use the full export. Then:

```
export ASRS_DIR=/path/to/your/csvs
export EMB_DIR=/path/to/glove      # glove.6B.*.txt from nlp.stanford.edu/projects/glove
cd asrs_pipeline
python3 corpus_stats.py            # prints your corpus counts next to ours
```

Within the task window the counts agree to within 0.1%.

## asrs_pipeline/: the Aircraft task

```
cd asrs_pipeline
bash run_all.sh
```

Runs, in order: corpus_stats.py, build_task.py (Aircraft task, fixed 80/20
split), train_word2vec.py, train_fasttext.py, run_queue.py and
run_queue_architectures.py (the GloVe, word2vec and fastText models),
report.py and paired_stats.py (bootstrap intervals and paired tests).
run_transformers.sh runs the contextual models; ablations.py runs the
token-order and 512-token-cap probes. Seeds are fixed; there is no
hyperparameter search. The full queue takes one GPU overnight. train.py and
train_transformers.py take --smoke for a two-minute run. Trainers write
per-record predictions; paired_stats.py is the only statistics step, so it
reruns without retraining.

## records/: the matched ASRS records

Same task, cases and split. Needs ASRS_DIR and the task built by
asrs_pipeline/build_task.py.

```
cd records
python3 build_records.py           # joins synopsis and supplemental narrative to the task
bash run_records.sh                # word2vec per record, BiLSTM finals, TF-IDF, paired statistics
python3 echo_mask.py               # category-vocabulary mask on both records
python3 matrix_2x2.py              # train and test across the two reporter narratives
python3 records_meanpool.py        # mean-pooling comparator
python3 interaction_test.py        # record-by-model interaction D
python3 control_glove_D.py         # D with one embedding shared across records
python3 records_charngram.py       # character n-gram baseline
python3 dual_report_length.py      # dual-report contrast by length of the second narrative
python3 dual_matched.py            # comparable-length rows per training
python3 review_budget.py           # error capture against review budget
python3 sensitivity_draws.py 0         # builds the negative-draw pool from the export
python3 sensitivity_draws.py <draw 1..5>          # one redrawn negative sample, one training
python3 sensitivity_extra.py <draw 1..5> <seed>   # trainings 2 and 3 per draw
python3 control_roberta.py         # RoBERTa: NHTSA fields and the synopsis, three seeds
python3 control_roberta_narr.py <seed> # RoBERTa: one narrative fine-tune per process
python3 control_roberta_stats.py   # paired contrasts over the RoBERTa predictions
python3 control_roberta_declared.py    # the three RoBERTa contrasts declared in advance
python3 seedavg_and_threshold.py   # Table 2 rows; needs the RoBERTa and NHTSA runs
```

records_task.jsonl.gz (53 MB) and the negative-draw pool (107 MB) are not
shipped; build_records.py and sensitivity_draws.py rebuild them from the
export. control_roberta_narr.py runs one fine-tune per process because
twelve in one process exceed 16 GB of memory.

## nhtsa/: recall campaigns

nhtsa_campaigns.jsonl.gz is the snapshot used: 18504 campaigns with
campaign numbers 2000 to 2026, retrieved August 2026; SHA256 in
SNAPSHOT_SHA256.txt; held-out campaign numbers in nhtsa_test_campaigns.txt.
The 16626 campaigns in the sixteen component classes with at least 300
campaigns form the task. nhtsa_crawl.py rebuilds the snapshot from the
public API (hours; resumable).

```
cd nhtsa
gunzip -k nhtsa_campaigns.jsonl.gz
python3 nhtsa_task.py              # 16-class task, exact duplicates confined to one side, TF-IDF and BiLSTM per field
python3 nhtsa_mask.py              # the same with class-label tokens masked
python3 control_nhtsa_shared.py    # one embedding shared across the three fields
python3 nhtsa_temporal.py          # train 2000-2021, test 2022-2026
python3 nhtsa_temporal_preds.py    # the temporal run with predictions kept, for the Holm-corrected contrasts
python3 nhtsa_neardup.py           # near-duplicate grouped split, Jaccard >= 0.80 in any field
python3 tabulate_neardup.py        # prints the near-duplicate tables and checks the criterion
```

The near-duplicate split's largest group has 4418 campaigns. Its held-out
set is a different sample from the reference split, so compare field
orderings, not absolute scores.

## ge_package/

The code run inside GE Aerospace; see ge_package/README.md. results/ge/
holds held-out macro-F1 per configuration (ge_results.jsonl), per-class
statistics and paired tests (ge_stats.json), the outcome-term audit and
keyword rule (leakage_report.json), the duplicate-grouped splits
(ge_dup_splits.json), training stability, strata and TF-IDF baselines.
ge_results.jsonl also carries rows from the initial implementation: the
10-fold means of Table S8 and the transformer trained from scratch. That
code and the training-stability summary are not part of this package.

## Scripts and outputs

Scripts write their JSON next to themselves; the copies under results/ are
what the paper used. Prediction files (.npz) are gitignored except the
near-duplicate set under results/nhtsa/neardup/.

Numbering follows the submitted manuscript and its supplement.

| paper | script |
|---|---|
| Table 2, Figure 4 (GE rows) | ge_package/ge_queue.py, ge_stats.py |
| Table 2 (ASRS and NHTSA rows) | records/seedavg_and_threshold.py |
| Figure 5, Tables S15, S16 | nhtsa/nhtsa_task.py, nhtsa_mask.py, control_nhtsa_shared.py |
| Figure 6, Table S9 | records/records_train.py, records_tfidf.py, records_charngram.py, records_stats.py, records_meanpool.py, control_roberta.py, dual_report_length.py |
| Tables S4, S5, Figures S2, S4, S5 | ge_package/ (ge_baselines.py, ge_train.py, ge_strata.py, ge_leakage.py, ge_covbound.py) |
| Tables S6, S7, Figure S1 | asrs_pipeline/train.py, train_transformers.py, paired_stats.py, report.py |
| Table S10 | records/records_meanpool.py, interaction_test.py, control_glove_D.py |
| Tables S11, S13 | records/records_train.py (dual-report scoring), records_stats.py, dual_report_length.py, dual_matched.py, seedavg_and_threshold.py |
| Table S12 | records/matrix_2x2.py |
| Table S14 | records/sensitivity_draws.py, sensitivity_extra.py |
| Table S17, Figure S3 | nhtsa/nhtsa_temporal.py, nhtsa_temporal_preds.py |
| Table S18 | nhtsa/nhtsa_neardup.py, tabulate_neardup.py |
| Tables S19, S20 | records/control_roberta.py, control_roberta_narr.py, control_roberta_stats.py, control_roberta_declared.py |
| Table S21, Figure S6 | records/review_budget.py |
| Section S1 record statistics | records/dual_report_length.py |

## Registration

protocol_maintnet.md was written and committed before any public data was
downloaded. MaintNet failed its go or no-go criteria; the addendum in the
same file registered the ASRS matched records and the NHTSA task.
registrations.md records the analyses declared later, each with the date
written, the criterion fixed in advance and the outcome. Those entries were
written in our working folder before each run and copied here afterwards;
the git timestamps of this repository are later than the runs.

| analysis | dataset | status in the paper |
|---|---|---|
| narrative vs synopsis | ASRS | preplanned |
| primary vs supplemental, raw transfer | ASRS | preplanned |
| length stratification and matched subset | ASRS | post hoc |
| dual-report train/test matrix | ASRS | post hoc |
| category-vocabulary mask | ASRS | post hoc |
| TF-IDF, character n-gram and pooling comparators | ASRS | post hoc |
| combination and disagreement review budget | ASRS | post hoc |
| interaction contrast D | ASRS | post hoc |
| shared-embedding control for D | ASRS | post hoc |
| five redrawn negative samples | ASRS | prospective extension |
| RoBERTa fine-tunes and three declared contrasts | ASRS, NHTSA | prospective extension |
| field hierarchy (summary, consequence, remedy) | NHTSA | preplanned |
| any-field duplicate confinement rerun | NHTSA | post hoc |
| class-vocabulary mask | NHTSA | post hoc |
| shared-embedding field control | NHTSA | post hoc |
| temporal split | NHTSA | prospective extension |
| near-duplicate grouped split | NHTSA | prospective extension |
| MaintNet phase 0 | MaintNet | registered, no-go |

The labels are those of Table S22 in the supplement.

## Reproducibility limits

Training on Apple MPS is not bit-reproducible. The temporal split was run
twice, the first time without saving predictions; the two runs agree in
direction on every contrast and differ in the third decimal. The NHTSA
snapshot and its hash ship because the database is updated continually.
The ASRS export grows month by month; within the task window it reproduces
our counts to within 0.1%. Intervals come from resampling the held-out
cases, and ranges over trainings are reported separately, so no inference
in the paper depends on a run being bit-identical.

## How to cite

Ihshaish, H., Mayhew, P., Zayet, T. M. A. and Del Amo, A. The record is part
of the task: matched-record evaluation of text classifiers across
maintenance, safety and recall reporting. Submitted to Computers in
Industry, 2026. Code and results: https://github.com/ihshaish/source-variation.
