#!/bin/zsh
until grep -q "queue2 complete" l1_queue2.log 2>/dev/null; do sleep 600; done
python3 l1_report.py > results/l1_report_snapshot.txt 2>&1
python3 l1_stats.py > results/l1_stats_snapshot.txt 2>&1
