#!/bin/zsh
cd "$(dirname "$0")"
export ASRS_DIR=/Users/Hisham/github_page/PhD_peter/data_asrs_v2
export L0_OUT=l0_out_v2 L1_DATA=l1_data_v2 L1_RES=results_v2 MAX_YEAR=2021
mkdir -p l1_data_v2 results_v2
python3 l0_corpus_stats.py > l0_v2.log 2>&1
python3 l1_build_task.py > l1_build_v2.log 2>&1
python3 l2_fasttext_train.py > l2_ft_v2.log 2>&1
python3 l2b_w2v_train.py > l2b_w2v_v2.log 2>&1
python3 l1_queue.py >> l1_queue_v2.log 2>&1
python3 l1_queue2.py > l1_queue2_v2.log 2>&1
python3 l1_train.py --emb w2vasrs --arch bilstm >> l1_queue3_v2.log 2>&1
python3 l1_train.py --emb w2vasrs --arch bigru >> l1_queue3_v2.log 2>&1
echo "v2 chain complete" >> l1_queue3_v2.log
python3 l1_report.py > results_v2/l1_report_snapshot.txt 2>&1
python3 l1_stats.py > results_v2/l1_stats_snapshot.txt 2>&1
