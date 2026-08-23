# paper_a_ge — GE-side experiments for the journal article

This folder contains the GE-side analysis pipeline for the paper. It is
self-contained: the scripts implement the same setup as the article's
public-corpus section (frozen embeddings, 10-fold cross-validation inside an
80% training partition, three seeded final models scored on the held-out 20%),
so results from here and from the public corpus are directly comparable. A
standard PC is sufficient; Python 3.10 or later, with or without a GPU (CPU
training is slower but workable overnight).

## Setup

Dependencies install with `python -m pip install -r requirements.txt`.

`python ge_selftest.py` generates a small synthetic dataset and runs the whole
chain on it, ending with SELFTEST PASSED when the environment is able to run
everything. It involves no real data and takes a couple of minutes; it is the
quickest way to confirm the machine is ready before the actual records are
exported.

## Inputs

Three inputs go into `ge_data\`:

- `ge_records.csv` — one row per repair record, columns
  `record_id, date, customer, technician, repair, label, unit_serial, operator`.
  Dates as YYYYMM or YYYY-MM-DD. Labels either 0–3 directly, or class names
  accompanied by a `label_map.json` of the form `{"Processor assembly": 0, ...}`.
  The `unit_serial` and `operator` columns can be partly empty but should be
  present: they are what makes the grouped robustness splits possible.
- `avi2vec.kv` — the Avi2Vec vectors, either as a gensim KeyedVectors save or
  in word2vec text/binary format; the loader accepts both.
- The GloVe files (`glove.6B.*.txt`) stay wherever they already are; the
  environment variable `EMB_DIR` points at that folder.

`lexicon.json` needs completing before the leakage audit: for each class, the
synonyms, abbreviations and part numbers a technician would plausibly write,
plus any replacement verbs missing from the seed list. This is a few minutes
of domain knowledge and determines what the audit counts.

## Running

The sequence is:

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

`ge_queue.py` carries the training matrix (roughly 250 model fits; minutes per
fit on a GPU, tens of minutes on CPU) and can be left unattended. Every result
is written under an idempotent key, so any interrupted script can be rerun and
will continue from where it stopped rather than repeating work.

## Outputs

The `results\` folder holds aggregate JSON files and per-record integer arrays
(true and predicted class indices); no narrative text is written anywhere in
it. That folder, together with the console output of `probe_terms.py`, is what
comes back for the paper.

## Coverage

One full run produces: the differential-vocabulary stratification and masking
test specified in the manuscript; the outcome-leakage audit with a keyword
baseline and a masked re-run; in-domain fastText and word2vec controls
matching the public-corpus pair; convolutional and mean-pooling architecture
probes; a character-n-gram TF-IDF baseline; duplicate-grouped, unit, operator
and temporal splits for the headline configuration; bootstrap confidence
intervals, paired randomisation tests, per-class tables and confusion
matrices; and the cosine-neighbour probe terms for the embedding table.

Outside the package's scope: the source-code checks on the original thesis
pipeline, transformer fine-tuning (a runner can be added once a suitable GE
machine is identified), and the human audit control arm.
