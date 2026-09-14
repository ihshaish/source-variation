# source-variation

Code and results for "The record is part of the task: matched-record
evaluation of text classifiers across maintenance, safety and recall
reporting" (submitted to Computers in Industry). One case, several records
of it, written by different people at different stages for different
purposes. The study measures how much classification performance moves
with the record a model reads, against how much representation and model
choice move it on the same cases, under the same labels and the same
split. Three systems: GE Aerospace repair events (customer report,
technician report, repair action; the label comes from the parts
transactions), NASA ASRS safety reports (reporter narrative, supplemental
narrative, analyst synopsis) and NHTSA recall campaigns (defect summary,
consequence, remedy).

If a number here disagrees with a number in the paper, or anything below
does not run for you, write to us:

- hisham.ihshaish@uwe.ac.uk
- peter.mayhew@geaerospace.com

## Short story

```
python3 reproduce.py
```

That reads results/ and prints every headline number in the paper next to
the value we shipped, 54 checks in all: the GE field levels and
differences, the ASRS reimplementation, the matched records, the NHTSA
field hierarchy, the seed-averaged Table 2 rows, the five redrawn negative
samples, the temporal and near-duplicate NHTSA splits and the RoBERTa
suite. No training, no downloads. If you see MISMATCH, something has
drifted; tell us. Retraining is the long story.

## Long story

### What runs here and what does not

Everything on the public side runs from a fresh clone plus two downloads
you make yourself: the ASRS export and the GloVe vectors. The NHTSA
snapshot ships in the repo. The GE side cannot run outside GE: the records
are proprietary and stay in the secure environment, and so do the Avi2Vec
vectors trained on them. What ships from GE is the code that ran there
(ge_package/), a self-test that drives the whole chain on invented
records, and the metrics that were exported (results/ge/). Nothing under
results/ contains record text.

### Getting the NASA data

The ASRS database is public: https://asrs.arc.nasa.gov/search/database.html.
Run a search over the date range you want (the paper uses reports through
2021 for the task and the full 1988 to 2026 export for corpus statistics),
export the results as CSV and put the files in one directory. The export
interface caps results per query, so you end up exporting year by year;
tedious but reliable. Then:

```
export ASRS_DIR=/path/to/your/csvs
cd asrs_pipeline
python3 corpus_stats.py     # checks your export against the paper's window
```

l0 prints the corpus counts next to ours; inside the task window they
agree to within 0.1%. Reports filed after our export date differ outside
the window, which is expected.

### asrs_pipeline/: the reimplementation

```
cd asrs_pipeline
python3 build_task.py       # Aircraft task, fixed 80/20 split
bash run_all.sh             # GloVe family, w2v and fastText, contextual
                               # models, order and cap probes
python3 paired_stats.py            # bootstrap intervals and paired tests
```

Seeds are fixed and there is no hyperparameter search; every constant was
fixed before a result was seen. The full queue is an overnight job on one
GPU. Every trainer takes --smoke for a two-minute sanity run; do that
before leaving it overnight. GloVe vectors download from
nlp.stanford.edu/projects/glove and the scripts look in EMB_DIR.
paired_stats.py is the only place statistics happen on this side; everything
upstream writes per-record predictions, so the statistics rerun without
retraining.

### views/: the matched ASRS records

Same Aircraft task, same cases, same split. build_records.py joins the
analyst synopsis and the supplemental narrative to the task by report
number from the same CSV export, then it is one queue:

```
cd views
python3 build_records.py         # needs ASRS_DIR, writes records_task.jsonl.gz
bash run_records.sh              # w2v per record, BiLSTM finals, TF-IDF, stats
python3 echo_mask.py           # category-vocabulary mask, both records
python3 matrix_2x2.py          # dual-report train/test matrix
python3 records_meanpool.py      # the pooling comparator
python3 interaction_test.py    # D = (sequence - pooling | synopsis) - (same | narrative)
python3 control_glove_D.py     # D again with one embedding shared across records
python3 seedavg_and_threshold.py   # Table 2: seed-averaged differences, joint intervals
```

The synopsis is the analyst's 19-token rewrite and it beats the 178-token
narrative, but only under sequence models, which is rather the point.
records_task.jsonl.gz (53 MB) is not shipped; build_records.py rebuilds it
from your export in a minute.

Two further ASRS runs live here. sensitivity_draws.py and
sensitivity_extra.py redraw the negative class five times (seeds 101 to
105) and retrain narrative and synopsis models on each draw; the pool file
they build from the export (107 MB) is not shipped either. The RoBERTa
suite is control_roberta.py (NHTSA fields and the synopsis),
control_roberta_narr.py (one narrative fine-tune per process, because
twelve in one process pushed a 16 GB laptop into swap),
control_roberta_stats.py (paired contrasts over the stored predictions)
and control_roberta_declared.py (the three contrasts we declared before the
ASRS runs finished).

### nhtsa/: recall campaigns

nhtsa_crawl.py enumerates recall campaigns by campaign number against the
public API (polite, resumable, takes hours). Or skip the crawl:
nhtsa_campaigns.jsonl.gz is the exact snapshot the paper used, 16626
campaigns after de-duplication, campaign numbers 2000 to 2026, retrieved
August 2026, SHA256 in SNAPSHOT_SHA256.txt, held-out campaign numbers in
nhtsa_test_campaigns.txt. The scripts read the uncompressed file, so
gunzip it first (keep the .gz for the hash).

```
cd nhtsa
gunzip -k nhtsa_campaigns.jsonl.gz
python3 nhtsa_task.py          # 16-class task, exact duplicates confined, TF-IDF and BiLSTM per field
python3 nhtsa_mask.py     # the same with every class-label token masked
python3 control_nhtsa_shared.py    # one embedding shared across the three fields
python3 nhtsa_temporal.py      # train 2000-2021, test 2022-2026
python3 nhtsa_temporal_preds.py    # the temporal rerun with predictions kept, for the Holm-corrected contrasts
python3 nhtsa_neardup.py       # near-duplicate grouped split, Jaccard >= 0.80 in any field
python3 tabulate_neardup.py    # Supplementary Table S18 and the registered criterion
```

nhtsa_leg.py is the first pass, kept for the record: its consequence-field
number was inflated by boilerplate duplicates straddling the split, which
is what leg2's any-field duplicate confinement fixes. The near-duplicate
run goes further and groups campaigns whose text in any field overlaps at
Jaccard 0.80 or more. Its largest group has 4418 campaigns, a transitive
union of generic consequence and remedy sentences rather than a set of
mutual duplicates; the ordering of the fields survives, the absolute
scores do not, and the two test sets are different samples, so compare
orderings and not levels.

### ge_package/

The code that ran inside GE Aerospace on the proprietary records.
ge_selftest.py drives the whole pipeline on invented data: same code
paths, fake records, useful for checking the logic and not the numbers.
Field statistics, the coverage bound, the masking conditions, the
alternative splits and the keyword baselines all live here. results/ge/
holds the metrics that left the environment: per-configuration held-out
macro-F1 (ge_results.jsonl), per-class statistics and paired tests
(ge_stats.json), the outcome-term audit and keyword rule
(leakage_report.json), seed stability, strata and TF-IDF baselines.

### What was registered, what was post hoc

protocol_maintnet.md is the registered note, written and committed before
we looked at any public data. The name is historical: MaintNet was the
first candidate dataset, it failed its own go/no-go, and the addendum in
the same file registered the ASRS matched records and the NHTSA legs. We
kept the filename because renaming a registration defeats the point.
CLAIMS_REGISTER_v5.1_controls.md carries the later declarations: the
shared-representation controls, the RoBERTa contrasts (declared while the
narrative fine-tunes were still training), the five negative redraws, the
temporal split and the near-duplicate split. Those addenda were written in
our working folder before each run and copied here afterwards, so the git
timestamps of this repository do not vouch for them; the file
modification times on our machines do, and we say so rather than pretend
otherwise.

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

Post hoc is not a dirty word here; everything above carries the same label
in the paper's supplement. The preplanned contrasts carry the confirmatory
weight, the rest explains them.

### What does not bit-reproduce, and why

Training on Apple MPS is not bit-reproducible run to run; the temporal
split was executed twice (the first run saved no predictions) and the two
executions agree in direction everywhere and differ in the third decimal.
The NHTSA crawl is a snapshot of a database that is updated continually,
which is why the snapshot and its hash ship. The ASRS export grows month
by month; inside the task window it reproduces our counts to within 0.1%.
Nothing in the paper's inference depends on a training run being
bit-identical: intervals come from resampling the held-out cases and
ranges over trainings are reported separately.

### Notes

The queue scripts are plain shell loops, not a scheduler, and assume one
job per GPU; the views and NHTSA queues are happy on a laptop with MPS or
CPU. Prediction files (.npz) are written next to the scripts and are
ignored by git except for the small near-duplicate set under
results/nhtsa/neardup/, which reproduce.py does not need but
tabulate_neardup.py does.

## How to cite

Ihshaish, H., Mayhew, P., Zayet, T. M. A. and Del Amo, A. The record is
part of the task: matched-record evaluation of text classifiers across
maintenance, safety and recall reporting. Submitted to Computers in
Industry, 2026. Code and results: https://github.com/ihshaish/source-variation.
