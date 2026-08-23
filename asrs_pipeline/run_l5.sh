#!/bin/zsh
cd "$(dirname "$0")"
python3 l5_contextual.py --model distilbert --mode frozen >> l5.log 2>&1
python3 l5_contextual.py --model safeaero --mode frozen >> l5.log 2>&1
python3 l5_contextual.py --model distilbert --mode finetune >> l5.log 2>&1
echo "l5 queue complete" >> l5.log
