# ge_package

The scripts run inside GE Aerospace on the proprietary repair records.
Setup as on the public side: frozen embeddings, 10-fold cross-validation
inside an 80% training partition, three trained models scored on the
held-out 20%. Python 3.10 or later; GPU optional.

## Setup

```
python -m pip install -r requirements.txt
python ge_selftest.py
```

The self-test runs the whole chain on a small invented dataset and ends
with SELFTEST PASSED. No real data; a couple of minutes.

## Inputs

Three inputs go into `ge_data/`:

- `ge_records.csv`, one row per repair record, with the columns
  `record_id, date, customer, technician, repair, label, unit_serial, operator`.
  Dates as DD/MM/YYYY, YYYY-MM-DD or YYYYMM. Labels either 0 to 3 directly, or class names
  with a `label_map.json` of the form `{"Processor assembly": 0, ...}`. `unit_serial`
  and `operator` may be partly empty; the grouped splits need them.
- `avi2vec.kv`, the Avi2Vec vectors, as a gensim KeyedVectors save or in
  word2vec text or binary format.
- The GloVe files (`glove.6B.*.txt`), wherever they already are; the
  environment variable `EMB_DIR` points at that folder.

`lexicon.json` lists, per class, the synonyms, abbreviations and part numbers
a technician would write, plus replacement verbs. Complete it before the
outcome-term audit; it determines what the audit counts.

## Running

```
python ge_build.py
python ge_embeds.py
python ge_strata.py --make-mask
python ge_leakage.py
python ge_queue.py
python ge_strata.py --analyse
python ge_baselines.py
python ge_stats.py
python probe_terms.py
```

`ge_queue.py` runs the training matrix: about 325 fits, minutes each on a
GPU, tens of minutes on CPU. Results are keyed, so an interrupted script
continues where it stopped.

The contrasts in `../ge_specification.md` run after that:

```
python ge_terms.py
python ge_naming_audit.py
python ge_tfidf_masked.py
python ge_queue_contrasts.py
python ge_contrasts.py
```

`ge_terms.py` lists the words most specific to each class, as a starting
point for completing `assembly_lexicon.json` with an engineer.
`ge_naming_audit.py` counts how often each field names each assembly and
writes the two masks. `ge_queue_contrasts.py` runs 22 configurations, three
trainings each, with `ge_train.py --no-cv`, which skips the ten-fold
cross-validation and gives the same held-out predictions as a full run.
`ge_contrasts.py` computes the paired intervals and tests. Output goes to
`results_contrasts/`.

## Outputs

`results/` and `results_contrasts/` hold aggregate JSON files and per-record
integer arrays of true and predicted class indices; no record text.
`results/` and the console output of `probe_terms.py` are what left GE, and
are results/ge/ in the repository root. `private/` holds word lists drawn
from the records and stays with them.

## What one run covers

- the differential-vocabulary stratification and masking test
- the outcome-term audit with a keyword baseline and a masked rerun
- in-domain fastText and word2vec controls matching the public-side pair
- convolutional and mean-pooling architecture probes
- a character n-gram TF-IDF baseline
- duplicate-grouped, unit, operator and temporal splits for the headline configuration
- bootstrap confidence intervals, paired randomisation tests, per-class tables and confusion matrices
- the cosine-neighbour probe terms for the embedding table

Not in the package: the transformer trained from scratch and the 10-fold
means of the initial implementation (earlier pipeline of the initial study).
Transformer fine-tuning was not run on the GE records; pretrained
checkpoints cannot be brought into the environment.

## Synthetic records and the container

The repair records cannot leave GE. `make_synthetic_ge.py` writes invented
records in the same input schema: 5,792 events with the published class
counts, customer, technician and repair-action texts of about the published
mean lengths, and assembly naming rates per class close to those of the
naming audit. The wording, the two assembly names, dates, units and
operators are made up, and the embedding file it writes holds random
vectors under the GloVe file name. The synthetic records show that the
scripts run from start to end. They do not reproduce the GE results.

`run_synthetic.sh` generates the records, builds the splits, trains the
three fields with a BiLSTM and with mean pooling (three trainings each,
`--no-cv`) and runs `paired_stats.py`: paired bootstrap intervals, unit
cluster intervals and Holm-adjusted randomisation tests for the field and
model differences. The Dockerfile pins Python 3.11.9, torch 2.3.1 (CPU),
NumPy 1.26.4 and scikit-learn 1.5.1:

```
docker build -t ge-synthetic ge_package
docker run --rm -v "$PWD/synthetic:/app/synthetic" ge-synthetic
docker run --rm ge-synthetic python bench_latency.py --threads 1
```

The workflow in `.github/workflows/container.yml` runs the same commands on
every change to this folder. Without Docker, install
`requirements_pinned.txt` and run `sh run_synthetic.sh`. Trainings differ
slightly between machines, so the numbers will not match digit for digit.

`bench_latency.py` counts the parameters of the four architectures and times
CPU inference at the three field length caps. Tokenising is not timed, and
the times show relative cost only.

## Your own records

The scripts take any CSV with one row per case, up to three text fields and
a label of up to four classes, in the schema above. Put it in place of the synthetic file, point
`EMB_DIR` at the GloVe vectors and run the steps of `run_synthetic.sh` after
the generator. `paired_stats.py` then reports the difference between fields
and between models with paired intervals.
