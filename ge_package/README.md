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
  Dates as YYYYMM or YYYY-MM-DD. Labels either 0 to 3 directly, or class names
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

## Outputs

`results/` holds aggregate JSON files and per-record integer arrays of true
and predicted class indices; no record text. That folder and the console
output of `probe_terms.py` are what left GE, and are results/ge/ in the
repository root.

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
