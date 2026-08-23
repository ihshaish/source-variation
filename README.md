# source-variation

Code and results for "Sources of performance variation in the classification
of avionics maintenance records" (submitted to Computers in Industry). One
repair event, three narratives, one label generated from the parts
transactions: the study measures how much classification performance moves
with the field you read, against what representation and model choice move.

Questions, problems, disagreements with a number: hisham.ihshaish@uwe.ac.uk

## Short story

```
python3 reproduce.py
```

That reads the shipped results and prints every headline ASRS number next to
the paper's claim. 14 checks, no training, a few seconds. Retraining from
scratch is the long story below.

## Getting the NASA data

The ASRS database is public: https://asrs.arc.nasa.gov/search/database.html
Run a search over the date range you want (the paper uses reports through
2021 for the task, and the full 1988-2026 export for corpus statistics),
export the results as CSV, and put the files in a directory. The export
interface caps results per query, so you will be exporting in year-sized
batches; tedious but reliable. Then:

```
export ASRS_DIR=/path/to/your/csvs
cd asrs_pipeline
python3 l0_corpus_stats.py     # checks your export against the paper's window
```

l0 prints the corpus counts next to the ones in the paper; inside the task
window they should agree to within 0.1%. Reports filed after our export date
will differ outside the window, which is fine and expected.

## Retraining everything

```
cd asrs_pipeline
python3 l1_build_task.py       # Aircraft task, fixed 80/20 split
bash run_all_v2.sh             # the full queue: GloVe family, w2v/fastText,
                               # contextual models, order and cap probes
python3 l1_stats.py            # bootstrap CIs + paired tests -> l1_stats.json
```

Seeds are fixed and there is no hyperparameter search; every choice was
fixed before any result was seen. The full queue is an overnight job on one
GPU. Every trainer takes --smoke for a two-minute sanity run first; do that
before leaving it overnight. GloVe vectors download from the usual place
(nlp.stanford.edu/projects/glove); the scripts look in EMB_DIR.

## ge_package/

The exact code that ran inside GE Aerospace on the proprietary records. The
records cannot leave (proprietary and export-controlled), so ge_selftest.py
drives the whole pipeline on invented data instead: same code paths, fake
records, useful for checking the logic rather than the numbers. Field
statistics, the coverage bound, the masking conditions, the alternative
splits and the keyword baselines all live here.

## What does not ship

The GE records (not ours to give), the Avi2Vec vectors (proprietary, which
is why the public side of the paper deliberately uses only artefacts you can
download), and trained checkpoints (large, and everything retrains from the
scripts above).

## Notes

The queue scripts are plain shell loops, not a scheduler, and assume one job
per GPU. l1_stats.py is the only place statistics happen; everything
upstream just writes per-record predictions, so you can re-run the stats
without retraining. If a number here and a number in the paper ever
disagree, trust results/ and write to us.
