"""Trains a fastText embedding on the training narratives of the Aircraft task.

Only records outside the test split are used, so the embedding never sees
held-out text. Skip-gram with subword information, 200 dimensions to match
GloVe-200, window 5, minimum count 3, ten epochs, fixed seed. The settings
are not tuned.

Reads task_aircraft.jsonl.gz and split.json from ASRS_DATA and saves the
model there as fasttext_asrs_train_200d.bin. Run: python3 train_fasttext.py
"""
import gzip
import json
import os
import re

from gensim.models import FastText

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
    model = FastText(vector_size=200, window=5, min_count=3, sg=1, epochs=10,
                     workers=8, seed=20260802)
    model.build_vocab(corpus_iterable=sentences)
    model.train(corpus_iterable=sentences, total_examples=len(sentences), epochs=10)
    out_path = os.path.join(DATA, "fasttext_asrs_train_200d.bin")
    model.save(out_path)
    print(f"saved {out_path}; vocab {len(model.wv)}")


if __name__ == "__main__":
    main()
