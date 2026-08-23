"""Train the in-domain fastText and word2vec controls on the GE TRAINING
partition (all three fields), mirroring the public-corpus pair: identical
hyperparameters, subwords on/off. Causal: never sees held-out text."""
import json, os
from gensim.models import FastText, Word2Vec
from ge_lib import DATA, load_records, record_tokens

split = json.load(open(os.path.join(DATA, "splits.json")))["random"]
test_ids = set(split)
sents = []
for r in load_records():
    if r["id"] in test_ids:
        continue
    for f in ("customer", "technician", "repair"):
        toks = record_tokens(r, f)
        if toks:
            sents.append(toks)
print(f"{len(sents)} field-texts for embedding training")
for name, cls in [("fasttext_ge", FastText), ("w2v_ge", Word2Vec)]:
    m = cls(vector_size=200, window=5, min_count=3, sg=1, epochs=10, workers=8,
            seed=20260802)
    m.build_vocab(corpus_iterable=sents)
    m.train(corpus_iterable=sents, total_examples=len(sents), epochs=10)
    m.save(os.path.join(DATA, f"{name}_200d.bin"))
    print(f"saved {name}: vocab {len(m.wv)}")
