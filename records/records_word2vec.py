"""Trains word2vec for one record type on the training partition of that
record type only: 200 dimensions, window 5, min_count 3, skip-gram, 10
epochs, fixed seed. Reads records_task.jsonl.gz and writes
w2v_<record>_200d.kv.
Run: python3 records_word2vec.py narr|syn"""
import gzip
import json
import re
import sys
from gensim.models import Word2Vec

TOKEN_RE = re.compile(r"[a-z][a-z0-9/-]+")
record = sys.argv[1]
sentences = []
with gzip.open('records_task.jsonl.gz', 'rt') as f:
    for line in f:
        row = json.loads(line)
        if row['split'] == 'train' and row[record].strip():
            sentences.append(TOKEN_RE.findall(row[record].lower()))
print(record, "train texts:", len(sentences))
model = Word2Vec(vector_size=200, window=5, min_count=3, sg=1, epochs=10, workers=8, seed=20260802)
model.build_vocab(sentences)
model.train(corpus_iterable=sentences, total_examples=len(sentences), epochs=10)
model.wv.save(f"w2v_{record}_200d.kv")
print("saved", f"w2v_{record}_200d.kv", "vocab", len(model.wv))
