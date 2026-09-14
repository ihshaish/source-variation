# source-variation

Code and results for "The record is part of the task: matched-record
evaluation of text classifiers across maintenance, safety and recall
reporting", submitted to Computers in Industry.

One case is documented more than once: by different people, at different
stages of a workflow, for different purposes. The study holds the cases, the
labels and the split fixed and changes only the record the classifier reads.
The effect of the record can then be compared with the effect of the
representation and of the model on the same footing. Three systems: GE
Aerospace repair events (customer report, technician report, repair action;
the label comes from the parts transactions), NASA ASRS safety reports
(reporter narrative, supplemental narrative, analyst synopsis) and NHTSA
recall campaigns (defect summary, consequence, remedy).

If a number here disagrees with a number in the paper, or something below
does not run for you, write to us: hisham.ihshaish@uwe.ac.uk or
peter.mayhew@geaerospace.com.

## Checking the paper's numbers

```
python3 reproduce.py
```

This reads results/ and prints every headline number in the paper next to
the value stored here, 54 checks in all. It needs no training and no
downloads. A MISMATCH line means something has drifted; tell us.

## What runs and what does not

The ASRS and NHTSA sides run from a fresh clone plus two downloads: the ASRS
export and the GloVe vectors. The NHTSA snapshot ships in the repository.
The GE side cannot run outside GE: the records are proprietary and cannot be
shared, and the same holds for the Avi2Vec vectors trained on them. What
ships from GE is the code that ran there (ge_package/), a self-test that
drives the whole chain on invented records, and the metrics that were
exported (results/ge/). Nothing under results/ contains record text.

## The NASA data

The ASRS database is public: https://asrs.arc.nasa.gov/search/database.html.
Search the date range you want, export the results as CSV and put the files
in one directory. The paper uses reports through 2021 for the task and the
full 1988 to 2026 export for corpus statistics. The export interface caps
the number of results per query, so the export is done year by year. Then:

```
export ASRS_DIR=/path/to/your/csvs
cd asrs_pipeline
python3 corpus_stats.py
```

This prints the corpus counts next to ours. Inside the task window they
agree to within 0.1%. Reports filed after our export date differ outside the
window, as expected.

## asrs_pipeline/: the Aircraft task

```
cd asrs_pipeline
bash run_all.sh
```

run_all.sh runs the chain in order: corpus_stats.py, build_task.py, the
word2vec and fastText training, the queue of GloVe, word2vec and fastText
models, then report.py and paired_stats.py. build_task.py builds the Aircraft
task with a fixed 80/20 split. paired_stats.py computes the bootstrap
intervals and paired tests.
run_transformers.sh runs the contextual models and ablations.py the order and
cap probes. Seeds are fixed and there is no hyperparameter search;
every constant was set before a result was seen. The full queue is an
overnight job on one GPU. train.py and train_transformers.py take --smoke for a
two-minute run; do that before leaving the queue overnight. GloVe vectors download from
nlp.stanford.edu/projects/glove and the scripts look for them in EMB_DIR.
paired_stats.py is the only place statistics happen on this side; everything
upstream writes per-record predictions, so the statistics rerun without
retraining.

## records/: the matched ASRS records

Same Aircraft task, same cases, same split. build_records.py joins the
analyst synopsis and the supplemental narrative to the task by report number
from the same CSV export; it needs ASRS_DIR and finds the task in
asrs_pipeline/data. Then:

```
cd records
python3 build_records.py
bash run_records.sh
python3 echo_mask.py
python3 matrix_2x2.py
python3 records_meanpool.py
python3 interaction_test.py
python3 control_glove_D.py
python3 records_charngram.py
python3 dual_report_length.py
python3 dual_matched.py
python3 review_budget.py
```

run_records.sh trains word2vec per record, the BiLSTM finals, the TF-IDF
baselines and runs the paired statistics. echo_mask.py applies the
category-vocabulary mask to both records. matrix_2x2.py trains and tests
across the two reporter narratives. records_meanpool.py is the mean-pooling
comparator, and interaction_test.py computes the record-by-model interaction
D from its predictions and the BiLSTM's. control_glove_D.py computes D again
with one embedding shared across the two records. records_charngram.py is
the character n-gram baseline. The synopsis is the analyst's 19-token rewrite
of a 178-token narrative, and the sequence models score higher on it.
records_task.jsonl.gz (53 MB) is not shipped; build_records.py rebuilds it
from your export in a minute.

Two further ASRS runs live here. sensitivity_draws.py and
sensitivity_extra.py redraw the negative class five times (seeds 101 to 105)
and retrain the narrative and synopsis models on each draw. The pool file
they build from the export (107 MB) is not shipped either. The RoBERTa
fine-tunes are control_roberta.py (the NHTSA fields and the synopsis) and
control_roberta_narr.py (one narrative fine-tune per process, because twelve
in one process ran out of memory on a 16 GB machine). control_roberta_stats.py
runs the paired contrasts over the stored predictions and
control_roberta_declared.py the three contrasts we declared before the ASRS
runs finished. Once the RoBERTa and NHTSA runs exist, seedavg_and_threshold.py
produces the rows of Table 2: differences averaged over trainings with joint
record-resampling intervals. Three small scripts read the stored predictions only:
dual_report_length.py bins the dual-report cases by the length of the second
narrative, dual_matched.py gives the comparable-length rows per training, and
review_budget.py computes error capture against review budget.

## nhtsa/: recall campaigns

nhtsa_crawl.py enumerates recall campaigns by campaign number against the
public API. It waits between requests, can be stopped and resumed, and takes hours. You can skip it.
nhtsa_campaigns.jsonl.gz is the snapshot the paper used: 18504 campaigns
with campaign numbers from 2000 to 2026, retrieved in August 2026. The
16626 of them in the sixteen component classes with at least 300 campaigns
form the task. The SHA256 of the snapshot is in SNAPSHOT_SHA256.txt and the
held-out campaign numbers are in nhtsa_test_campaigns.txt. The scripts read the
uncompressed file, so gunzip it first and keep the .gz for the hash.

```
cd nhtsa
gunzip -k nhtsa_campaigns.jsonl.gz
python3 nhtsa_task.py
python3 nhtsa_mask.py
python3 control_nhtsa_shared.py
python3 nhtsa_temporal.py
python3 nhtsa_temporal_preds.py
python3 nhtsa_neardup.py
python3 tabulate_neardup.py
```

nhtsa_task.py builds the 16-class task, confines exact duplicates to one side
of the split and trains TF-IDF and BiLSTM classifiers per field.
nhtsa_mask.py repeats this with every class-label token masked.
control_nhtsa_shared.py shares one embedding across the three fields.
nhtsa_temporal.py trains on campaigns filed 2000 to 2021 and tests on 2022
to 2026; nhtsa_temporal_preds.py is the same run with predictions kept, for
the Holm-corrected contrasts. nhtsa_neardup.py groups campaigns whose text in
any field overlaps at Jaccard 0.80 or more and confines each group to one
side. Its largest group has 4418 campaigns, a chain of generic consequence
and remedy sentences. The ordering of the fields holds under that split; the
absolute scores do not carry over, because the held-out campaigns are a
different sample. tabulate_neardup.py prints the tables and checks the
criterion fixed before the run.

## ge_package/

The code that ran inside GE Aerospace on the proprietary records.
ge_selftest.py drives the whole pipeline on invented data: the same code
paths, fake records, useful for checking the logic and not the numbers.
Field statistics, the coverage bound, the masking conditions, the alternative
splits and the keyword baselines all live here. results/ge/ holds the
metrics that were exported: held-out macro-F1 per configuration
(ge_results.jsonl), per-class statistics and paired tests (ge_stats.json),
the outcome-term audit and keyword rule (leakage_report.json), the
duplicate-grouped splits (ge_dup_splits.json), training stability, strata
and TF-IDF baselines. ge_results.jsonl also carries the rows of the initial
implementation: the 10-fold means of Table S8 and the transformer trained
from scratch. That code is the earlier pipeline of the initial study and is
not part of this package. The same holds for the training-stability summary.

## Which script produces which table

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

## What was registered and what was post hoc

protocol_maintnet.md is the registered note, written and committed before we
looked at any public data. The name is the one it was committed under:
MaintNet was the first candidate dataset, it failed its own go or no-go
criteria, and the addendum in the same file registered the ASRS matched
records and the NHTSA task. registrations.md carries the later declarations:
the shared-representation controls, the RoBERTa contrasts (written while the
narrative fine-tunes were still training), the five negative redraws, the
temporal split and the near-duplicate split, each with the criterion fixed in
advance and the outcome. Those entries were written in our working folder
before each run and copied here afterwards, so the git timestamps here are
later than the runs. The modification times in our working folder are the
earlier ones.

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

The labels are those of the paper's supplement (Table S22), which also
lists the GE analyses. The preplanned contrasts carry the confirmatory
weight; the rest explains them.

## What does not reproduce bit for bit

Training on Apple MPS is not bit-reproducible from run to run. The temporal
split was executed twice, once without saving predictions, and the two
executions agree in direction everywhere and differ in the third decimal.
The NHTSA crawl is a snapshot of a database that is updated continually,
which is why the snapshot and its hash ship. The ASRS export grows month by
month; inside the task window it reproduces our counts to within 0.1%.
Nothing in the paper's inference depends on a training run being
bit-identical: intervals come from resampling the held-out cases, and ranges
over trainings are reported separately.

## Notes

The queue scripts are plain shell loops, one job per GPU. The records and
NHTSA queues run on a laptop with MPS or CPU. Prediction files (.npz) are
written next to the scripts and are ignored by git, except the small
near-duplicate set under results/nhtsa/neardup/, which tabulate_neardup.py
reads.

## How to cite

Ihshaish, H., Mayhew, P., Zayet, T. M. A. and Del Amo, A. The record is part
of the task: matched-record evaluation of text classifiers across
maintenance, safety and recall reporting. Submitted to Computers in
Industry, 2026. Code and results: https://github.com/ihshaish/source-variation.
