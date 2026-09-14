# ge_package

The GE side of the study. These scripts ran inside GE Aerospace on the
proprietary repair records and implement the same setup as the public side:
frozen embeddings, 10-fold cross-validation inside an 80% training partition,
three trained models scored on the held-out 20%. A standard PC is enough;
Python 3.10 or later, with or without a GPU.

## Setup

```
python -m pip install -r requirements.txt
python ge_selftest.py
```

The self-test generates a small invented dataset and runs the whole chain on
it, ending with SELFTEST PASSED. It touches no real data and takes a couple
of minutes.

## Inputs

Three inputs go into `ge_data/`:

- `ge_records.csv`, one row per repair record, with the columns
  `record_id, date, customer, technician, repair, label, unit_serial, operator`.
  Dates as YYYYMM or YYYY-MM-DD. Labels either 0 to 3 directly, or class names
  with a `label_map.json` of the form `{"Processor assembly": 0, ...}`. The
  `unit_serial` and `operator` columns may be partly empty but should exist;
  they make the grouped splits possible.
- `avi2vec.kv`, the Avi2Vec vectors, as a gensim KeyedVectors save or in
  word2vec text or binary format.
- The GloVe files (`glove.6B.*.txt`), wherever they already are; the
  environment variable `EMB_DIR` points at that folder.

`lexicon.json` lists, per class, the synonyms, abbreviations and part numbers
a technician would write, plus replacement verbs. It has to be completed
before the outcome-term audit, because it determines what the audit counts.

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

`ge_queue.py` runs the training matrix, about 325 model fits, minutes per fit
on a GPU and tens of minutes on CPU, and can be left unattended. Every result
is written under a key, so an interrupted script continues from where it
stopped when rerun.

## Outputs

The `results/` folder holds aggregate JSON files and per-record integer arrays
of true and predicted class indices. No record text is written anywhere in it.
That folder, with the console output of `probe_terms.py`, is what left GE for
the paper and is what results/ge/ in the repository root contains.

## What one run covers

The differential-vocabulary stratification and masking test; the outcome-term
audit with a keyword baseline and a masked rerun; in-domain fastText and
word2vec controls matching the public-side pair; convolutional and
mean-pooling architecture probes; a character n-gram TF-IDF baseline;
duplicate-grouped, unit, operator and temporal splits for the headline
configuration; bootstrap confidence intervals, paired randomisation tests,
per-class tables and confusion matrices; and the cosine-neighbour probe terms
for the embedding table. Two things are outside the package. The transformer trained from scratch
and the 10-fold means of the initial implementation come from the earlier
pipeline of the initial study. Transformer fine-tuning was not run on the
GE records, because pretrained checkpoints cannot be brought into the
environment.
