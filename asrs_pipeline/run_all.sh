#!/bin/zsh
cd "$(dirname "$0")"
export ASRS_DIR=${ASRS_DIR:-../data_asrs_v2}   # point this at your ASRS CSV export
export CORPUS_OUT=corpus_out_v2 ASRS_DATA=data_v2 ASRS_RES=results_v2 MAX_YEAR=2021
mkdir -p data_v2 results_v2
python3 corpus_stats.py > l0_v2.log 2>&1
python3 build_task.py > l1_build_v2.log 2>&1
python3 train_fasttext.py > l2_ft_v2.log 2>&1
python3 train_word2vec.py > l2b_w2v_v2.log 2>&1
python3 run_queue.py >> run_queue_v2.log 2>&1
python3 run_queue_architectures.py > run_queue_architectures_v2.log 2>&1
python3 train.py --emb w2vasrs --arch bilstm >> run_queue3_v2.log 2>&1
python3 train.py --emb w2vasrs --arch bigru >> run_queue3_v2.log 2>&1
echo "v2 chain complete" >> run_queue3_v2.log
python3 report.py > results_v2/report_snapshot.txt 2>&1
python3 paired_stats.py > results_v2/asrs_stats_snapshot.txt 2>&1
