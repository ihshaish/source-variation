#!/bin/zsh
cd "$(dirname "$0")"
python3 nhtsa_crawl.py 0 17
python3 nhtsa_crawl.py 21 26
python3 nhtsa_leg.py
touch LEGB_DONE
