"""Word2vec per view, trained on the training partition of that view only.
Same params as the paper's l2b trainer (200d, window 5, min_count 3, skip-gram,
10 epochs, fixed seed)."""
import gzip, json, re, sys
from gensim.models import Word2Vec
TOKEN_RE = re.compile(r"[a-z][a-z0-9/-]+")
view = sys.argv[1]  # narr | syn
sents=[]
with gzip.open('views_task.jsonl.gz','rt') as f:
    for l in f:
        r=json.loads(l)
        if r['split']=='train' and r[view].strip():
            sents.append(TOKEN_RE.findall(r[view].lower()))
print(view,"train texts:",len(sents))
m=Word2Vec(vector_size=200, window=5, min_count=3, sg=1, epochs=10, workers=8, seed=20260802)
m.build_vocab(sents)
m.train(corpus_iterable=sents, total_examples=len(sents), epochs=10)
m.wv.save(f"w2v_{view}_200d.kv")
print("saved", f"w2v_{view}_200d.kv", "vocab", len(m.wv))
