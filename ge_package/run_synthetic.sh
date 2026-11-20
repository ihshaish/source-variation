#!/bin/sh
# Generates the synthetic records, then runs the build, six configurations
# (three fields, BiLSTM and mean pooling, three trainings each) and the
# paired statistics. Output goes to synthetic/ next to this script.
set -e
cd "$(dirname "$0")"
export GE_DATA="$PWD/synthetic/ge_data"
export GE_RES="$PWD/synthetic/results"
export EMB_DIR="$PWD/synthetic/emb"
# ge_build.py hashes text for the near-duplicate split, so the hash seed is fixed
export PYTHONHASHSEED=0
rm -rf synthetic/ge_data synthetic/results synthetic/emb

python make_synthetic_ge.py
python ge_build.py
for field in customer technician repair; do
    for arch in bilstm meanmlp; do
        python ge_train.py --field "$field" --emb glove200 --arch "$arch" --no-cv
    done
done
python paired_stats.py
