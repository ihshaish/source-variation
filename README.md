# matched-records

Code and public-corpus reproduction for "Sources of performance variation in
the classification of avionics maintenance records" (submitted to Computers in
Industry). One repair event, three narratives, one label generated from the
parts transactions: the code measures how much classification performance
moves with the field you read, compared with what representation and model
choice move.

## Short story

```
cd asrs_pipeline
python3 download_v2.py        # public NASA ASRS export, resumes if interrupted
python3 l1_build_task.py      # Aircraft task, fixed 80/20 split
bash run_all_v2.sh            # every ASRS configuration in the paper
python3 l1_stats.py           # bootstrap CIs, paired randomisation tests, Holm
```

That reproduces the ASRS side of the paper end to end from public data. Seeds
are fixed, there is no hyperparameter search, and every choice was fixed
before any result was seen. On one GPU the full queue is an overnight job;
`--smoke` on any trainer gives a two-minute sanity run first.

## Long story

`asrs_pipeline/` is staged the way the experiments ran: l0 downloads and
reconciles the corpus against the paper's 1989-2021 window (it prints the
count differences, which should be within 0.1%), l1 builds the task and
trains the GloVe family, l2 trains the word2vec and fastText comparators on
the task partition, l5 runs the frozen and fine-tuned contextual models, l6
runs the shuffled-order and 512-token probes. l1_stats.py is the only place
statistics happen; everything upstream just writes per-record predictions.

`ge_package/` is the exact code that ran inside GE Aerospace on the
proprietary records. The records cannot leave (proprietary and
export-controlled), so `ge_selftest.py` drives the whole pipeline on
synthetic records instead: same code paths, invented data, useful for
checking the logic rather than the numbers. The field statistics, the
coverage bound, the masking conditions, the alternative splits and the
keyword baselines are all here.

`results/` holds the shipped ASRS outputs the paper's tables read
(`l1_stats.json` is the file the numbers come from), so the statistics stage
can be re-run without retraining anything.

## What does not ship

The GE records (not ours to give), the Avi2Vec vectors (proprietary, which is
why the public side of the paper deliberately uses only artefacts you can
download), and trained checkpoints (large, and everything retrains from the
scripts above).

## Notes

The ASRS export grows over time, so a fresh download will not match the
paper's corpus exactly outside the task window; l0 checks the window that
matters and reports what it finds. The queue scripts are plain shell loops,
not a scheduler, and they assume one job per GPU. If a number in the paper
and a number in `results/` ever disagree, trust `results/` and tell us.
