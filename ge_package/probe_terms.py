"""P7 - cosine-neighbour probe terms for the manuscript's neighbours table.
The TERMS list below is edited to taste; usage: python probe_terms.py
Prints, per term: top-10 Avi2Vec neighbours + whether the term exists in the
GloVe-200 vocabulary (and its GloVe neighbours where it does, if the full
GloVe .txt is reachable via EMB_DIR)."""
import os
from gensim.models import KeyedVectors
from ge_lib import DATA

TERMS = ["underfill", "keypanel", "blank", "flickering"]  # add a fault code etc.

kv = None
path = os.path.join(DATA, "avi2vec.kv")
try:
    kv = KeyedVectors.load(path, mmap="r")
except Exception:
    kv = KeyedVectors.load_word2vec_format(path, binary=path.endswith(".bin"))
emb_dir = os.environ.get("EMB_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "embeddings"))
glove_vocab = set()
gpath = os.path.join(emb_dir, "glove.6B.200d.txt")
if os.path.exists(gpath):
    with open(gpath, encoding="utf-8") as f:
        for line in f:
            glove_vocab.add(line.split(" ", 1)[0])
for t in TERMS:
    print(f"\n== {t} ==")
    if t in kv:
        for w, s in kv.most_similar(t, topn=10):
            print(f"  {w:<20} {s:.2f}")
    else:
        print("  (not in Avi2Vec vocabulary)")
    print(f"  in GloVe-200 vocabulary: {t in glove_vocab if glove_vocab else 'unknown (GloVe file not found)'}")
