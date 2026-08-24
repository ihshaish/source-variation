# source-variation

Code and results for "Sources of performance variation in operational text
classification" (submitted to Computers in Industry). One operational case,
several records of it: the study measures how much classification performance
moves with the record you read, against what representation and model choice
move. Three systems: GE repair events (three narratives, one label from the
parts transactions), ASRS safety reports (reporter narrative, supplemental
narrative, analyst synopsis), NHTSA recalls (defect summary, consequence,
remedy).

If you have any questions, or come across a problem...or have any disagreements with a number, please let us know: 
- hisham.ihshaish@uwe.ac.uk
- peter.mayhew@geaerospace.com

## Short story

```
python3 reproduce.py
```

That reads the reported results & prints every headline public-data number
next to the paper's claim basically - 29 checks now (14 for the ASRS
reimplementation, 12 for the matched views and NHTSA, 3 for the
shared-representation controls). No training needed.
Retraining from scratch is the long story below.

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

## What is not shown/shipped;

The GE records (these are GE's), the Avi2Vec vectors (proprietary... which
is why the public side of the paper uses only artefacts you can
download), trained checkpoints (large, & everything retrains from the
scripts above), and views_task.jsonl.gz (53MB; build_views.py rebuilds it
from your own ASRS export in a minute).

## views/

The matched-view experiments on the same Aircraft task (same cases, same
split - build_views.py joins synopsis and supplemental narrative to the task
by report number, from the same CSV export as above). Then it is one queue:

```
cd views
python3 build_views.py         # needs ASRS_DIR, writes views_task.jsonl.gz
bash run_views.sh              # w2v per view -> BiLSTM finals -> tfidf -> stats
python3 echo_mask.py           # taxonomy-token mask, both views
python3 matrix_2x2.py          # dual-report train/test matrix
python3 meanpool_views.py      # the non-sequence control
python3 interaction_test.py    # D = (seq-pool | syn) - (seq-pool | narr)
```

The synopsis is the analyst's 19-token rewrite and it beats the 178-token
narrative - but only under sequence models, which is rather the point.

## nhtsa/

The second public system. nhtsa_crawl.py enumerates recall campaigns by
campaign number against the public API (polite, resumable, takes hours),
or skip the crawl: nhtsa_campaigns.jsonl.gz is the exact snapshot the paper
used (16626 campaigns after de-dup, campaign numbers 2000-2026, retrieved
August 2026; SHA256 in SNAPSHOT_SHA256.txt, held-out campaign numbers in
nhtsa_test_campaigns.txt - the split is also fully determined by seed
20260802 in the code). nhtsa_leg2.py builds the 16-class component task and
runs TF-IDF + BiLSTM per field; nhtsa_leg3_mask.py re-runs everything with
every token from the class labels masked out of training and evaluation.
nhtsa_leg.py is the first pass kept for the record - its consequence-field
number was inflated by boilerplate duplicates straddling the split, which is
exactly what leg2's any-field duplicate confinement fixes.

## What was registered, what was post hoc

protocol_maintnet.md is the registered note, written and committed before we
looked at any of this data - the name is historical (MaintNet was the first
candidate dataset; it failed its own pre-registered go/no-go and the addendum
in the same file registered the ASRS-views and NHTSA legs). We kept the
filename because renaming a registration defeats the point; the git
timestamps are the receipts.

| analysis | dataset | status |
|---|---|---|
| narrative vs synopsis | ASRS | registered |
| primary vs supplemental, raw | ASRS | registered |
| length stratification + matched subset | ASRS | post hoc |
| 2x2 dual train/test matrix | ASRS | post hoc |
| taxonomy-token mask | ASRS | post hoc |
| TF-IDF / char n-gram / mean-pool controls | ASRS | post hoc (controls) |
| ensemble + disagreement budget | ASRS | post hoc |
| interaction contrast D | ASRS | post hoc |
| shared-representation control for D (control_glove_D.py) | ASRS | post hoc |
| field hierarchy (summary/consequence/remedy) | NHTSA | registered |
| any-field duplicate confinement re-run | NHTSA | post hoc (hygiene) |
| class-vocabulary mask | NHTSA | post hoc |
| shared-representation field control (control_nhtsa_shared.py) | NHTSA | post hoc |
| MaintNet phase 0 | MaintNet | registered, no-go |

The two control_ scripts are the review-round controls: D re-estimated with
GloVe-200 fixed across both views (survives, 2/3 CIs exclude zero, and the
same runs showed the mean-pooled view contrast is specific to view-trained
embeddings - see CLAIMS_REGISTER_v5.1_controls.md), and the NHTSA hierarchy
under one shared word2vec (summary primacy survives, the consequence-remedy
ordering does not - it was partly the field-trained representations).

Post hoc is not a dirty word here - everything above is labelled the same
way in the paper. The registered contrasts carry the confirmatory weight;
the rest explains them.

## Notes

The queue scripts are plain shell loops, not a scheduler, and assume one job
per GPU (the views and NHTSA queues are happy on a laptop with MPS or CPU).
l1_stats.py and views/views_stats.py are the only places statistics happen;
everything upstream just writes per-record predictions, so you can re-run
the stats without retraining. If a number here and a number in the paper
ever disagree, trust results/ and write to us.
