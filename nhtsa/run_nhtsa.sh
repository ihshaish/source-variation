#!/bin/zsh
cd "$(dirname "$0")"
python3 nhtsa_crawl.py 0 26
python3 nhtsa_task.py
touch NHTSA_DONE
