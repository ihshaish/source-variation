"""L2 — train the subword (fastText) embedding on the TRAIN partition text only.

Causal by construction: the model never sees held-out text. 200d to match the
GloVe-200/Avi2Vec reference dimensionality. Skip-gram, standard hyperparameters;
these are stated in the paper, not tuned (the baseline is a representation
probe, not a tuned competitor).
"""
import gzip
import json
import os
DATA = os.environ.get("L1_DATA", None)
import re

from gensim.models import Word2Vec

HERE = os.path.dirname(__file__)
TOKEN_RE = re.compile(r"[a-z][a-z0-9/-]+")


def main():
    with open(os.path.join(DATA or os.path.join(HERE, "l1_data"), "split.json")) as f:
        test_acns = set(json.load(f)["test_acns"])
    sents = []
    with gzip.open(os.path.join(DATA or os.path.join(HERE, "l1_data"), "task_aircraft.jsonl.gz"), "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["acn"] in test_acns:
                continue
            sents.append(TOKEN_RE.findall(r["text"].lower()))
    print(f"{len(sents)} training narratives")
    model = Word2Vec(vector_size=200, window=5, min_count=3, sg=1, epochs=10,
                     workers=8, seed=20260802)
    model.build_vocab(corpus_iterable=sents)
    model.train(corpus_iterable=sents, total_examples=len(sents), epochs=10)
    out = os.path.join(DATA or os.path.join(HERE, "l1_data"), "w2v_asrs_train_200d.bin")
    model.save(out)
    print(f"saved {out}; vocab {len(model.wv)}")


if __name__ == "__main__":
    main()
