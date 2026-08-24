#!/bin/zsh
cd "$(dirname "$0")"
set -x
python3 views_w2v.py narr
python3 views_w2v.py syn
python3 views_train.py --view syn --emb glove200 --smoke
python3 views_train.py --view syn --emb glove200
python3 views_train.py --view syn --emb w2vview
python3 views_train.py --view narr --emb glove200
python3 views_train.py --view narr --emb w2vview
python3 views_tfidf.py
python3 views_stats.py
touch QUEUE_DONE
