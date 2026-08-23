#!/bin/zsh
until grep -q "queue2 complete" l1_queue2.log 2>/dev/null; do sleep 600; done
python3 l1_train.py --emb w2vasrs --arch bilstm >> l1_queue3.log 2>&1
python3 l1_train.py --emb w2vasrs --arch bigru >> l1_queue3.log 2>&1
echo "queue3 complete" >> l1_queue3.log
python3 l1_report.py > results/l1_report_snapshot.txt 2>&1
python3 l1_stats.py > results/l1_stats_snapshot.txt 2>&1
