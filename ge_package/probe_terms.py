"""Prints, for each term in TERMS, its ten nearest Avi2Vec neighbours by
cosine similarity and whether the term is in the GloVe-200 vocabulary.
Reads ge_data/avi2vec.kv and, when EMB_DIR points at it, glove.6B.200d.txt.
Edit TERMS before running. Run: python probe_terms.py
"""
import os

from gensim.models import KeyedVectors

from ge_lib import DATA

TERMS = ["underfill", "keypanel", "blank", "flickering"]

avi2vec = None
path = os.path.join(DATA, "avi2vec.kv")
try:
    avi2vec = KeyedVectors.load(path, mmap="r")
except Exception:
    avi2vec = KeyedVectors.load_word2vec_format(path, binary=path.endswith(".bin"))
emb_dir = os.environ.get("EMB_DIR",
                         os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "embeddings"))
glove_vocab = set()
glove_path = os.path.join(emb_dir, "glove.6B.200d.txt")
if os.path.exists(glove_path):
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            glove_vocab.add(line.split(" ", 1)[0])
for term in TERMS:
    print(f"\n== {term} ==")
    if term in avi2vec:
        for neighbour, score in avi2vec.most_similar(term, topn=10):
            print(f"  {neighbour:<20} {score:.2f}")
    else:
        print("  (not in Avi2Vec vocabulary)")
    if glove_vocab:
        in_glove = term in glove_vocab
    else:
        in_glove = "unknown (GloVe file not found)"
    print(f"  in GloVe-200 vocabulary: {in_glove}")
