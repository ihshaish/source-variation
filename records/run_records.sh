#!/bin/zsh
cd "$(dirname "$0")"
set -x
python3 records_word2vec.py narr
python3 records_word2vec.py syn
python3 records_train.py --record syn --emb glove200 --smoke
python3 records_train.py --record syn --emb glove200
python3 records_train.py --record syn --emb w2vview
python3 records_train.py --record narr --emb glove200
python3 records_train.py --record narr --emb w2vview
python3 records_tfidf.py
python3 records_stats.py
touch QUEUE_DONE
