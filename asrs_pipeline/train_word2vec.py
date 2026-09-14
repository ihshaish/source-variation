"""Trains a word2vec embedding on the training narratives of the Aircraft task.

This is the no-subword counterpart of train_fasttext.py: the same records,
the same skip-gram settings (200 dimensions, window 5, minimum count 3, ten
epochs, fixed seed), but whole words only, so a word outside the training
vocabulary gets no vector. Only records outside the test split are used.

Reads task_aircraft.jsonl.gz and split.json from ASRS_DATA and saves the
model there as w2v_asrs_train_200d.bin. Run: python3 train_word2vec.py
"""
import gzip
import json
import os
import re

from gensim.models import Word2Vec

HERE = os.path.dirname(__file__)
DATA = os.environ.get("ASRS_DATA") or os.path.join(HERE, "data")
TOKEN_RE = re.compile(r"[a-z][a-z0-9/-]+")


def main():
    with open(os.path.join(DATA, "split.json")) as f:
        test_acns = set(json.load(f)["test_acns"])
    sentences = []
    with gzip.open(os.path.join(DATA, "task_aircraft.jsonl.gz"), "rt") as f:
        for line in f:
            record = json.loads(line)
            if record["acn"] in test_acns:
                continue
            sentences.append(TOKEN_RE.findall(record["text"].lower()))
    print(f"{len(sentences)} training narratives")
    model = Word2Vec(vector_size=200, window=5, min_count=3, sg=1, epochs=10,
                     workers=8, seed=20260802)
    model.build_vocab(corpus_iterable=sentences)
    model.train(corpus_iterable=sentences, total_examples=len(sentences), epochs=10)
    out_path = os.path.join(DATA, "w2v_asrs_train_200d.bin")
    model.save(out_path)
    print(f"saved {out_path}; vocab {len(model.wv)}")


if __name__ == "__main__":
    main()
