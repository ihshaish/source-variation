#!/bin/zsh
cd "$(dirname "$0")"
python3 train_transformers.py --model distilbert --mode frozen >> train_transformers.log 2>&1
python3 train_transformers.py --model safeaero --mode frozen >> train_transformers.log 2>&1
python3 train_transformers.py --model distilbert --mode finetune >> train_transformers.log 2>&1
echo "transformers queue complete" >> train_transformers.log
