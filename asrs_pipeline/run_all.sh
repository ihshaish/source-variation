#!/bin/zsh
cd "$(dirname "$0")"
export ASRS_DIR=${ASRS_DIR:?set ASRS_DIR to the folder holding the ASRS CSV export}
export CORPUS_OUT=corpus_out
export ASRS_DATA=data
export ASRS_RES=results
export MAX_YEAR=2021
mkdir -p data results
python3 corpus_stats.py > corpus_stats.log 2>&1
python3 build_task.py > build_task.log 2>&1
python3 train_fasttext.py > train_fasttext.log 2>&1
python3 train_word2vec.py > train_word2vec.log 2>&1
python3 run_queue.py >> run_queue.log 2>&1
python3 run_queue_architectures.py > run_queue_architectures.log 2>&1
python3 train.py --emb w2vasrs --arch bilstm >> train_w2vasrs.log 2>&1
python3 train.py --emb w2vasrs --arch bigru >> train_w2vasrs.log 2>&1
echo "chain complete" >> train_w2vasrs.log
python3 report.py > results/report_snapshot.txt 2>&1
python3 paired_stats.py > results/asrs_stats_snapshot.txt 2>&1
