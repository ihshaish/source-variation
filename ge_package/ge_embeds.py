"""Trains the in-domain fastText and word2vec vectors on the training
partition of the random split, using all three text fields, with the same
hyperparameters as the public-corpus pair (subwords on for fastText, off
for word2vec). Held-out text is never seen. Reads ge_records.jsonl.gz and
splits.json from GE_DATA and writes fasttext_ge_200d.bin and w2v_ge_200d.bin
there. Run: python ge_embeds.py
"""
import json
import os

from gensim.models import FastText, Word2Vec

from ge_lib import DATA, load_records, record_tokens

split = json.load(open(os.path.join(DATA, "splits.json")))["random"]
test_ids = set(split)
sentences = []
for record in load_records():
    if record["id"] in test_ids:
        continue
    for field in ("customer", "technician", "repair"):
        tokens = record_tokens(record, field)
        if tokens:
            sentences.append(tokens)
print(f"{len(sentences)} field-texts for embedding training")
for name, model_class in [("fasttext_ge", FastText), ("w2v_ge", Word2Vec)]:
    model = model_class(vector_size=200, window=5, min_count=3, sg=1, epochs=10, workers=8,
                        seed=20260802)
    model.build_vocab(corpus_iterable=sentences)
    model.train(corpus_iterable=sentences, total_examples=len(sentences), epochs=10)
    model.save(os.path.join(DATA, f"{name}_200d.bin"))
    print(f"saved {name}: vocab {len(model.wv)}")
