"""Runs train.py for each embedding and recurrent architecture in turn.

One configuration at a time, since there is a single GPU device. GloVe-200
and fastText come first, the other GloVe sizes after. Avi2Vec is added to
the end of the queue if data/avi2vec.kv exists. A configuration that fails
is reported and the queue moves on; train.py skips folds and seeds that are
already in the results file, so the queue can be restarted.
Run: python3 run_queue.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = [
    ("glove200", "bilstm"), ("glove200", "bigru"),
    ("fasttext", "bilstm"), ("fasttext", "bigru"),
    ("glove50", "bilstm"), ("glove50", "bigru"),
    ("glove100", "bilstm"), ("glove100", "bigru"),
    ("glove300", "bilstm"), ("glove300", "bigru"),
]
if os.path.exists(os.path.join(HERE, "data", "avi2vec.kv")):
    CONFIGS += [("avi2vec", "bilstm"), ("avi2vec", "bigru")]

for emb, arch in CONFIGS:
    print(f"=== {emb} / {arch} ===", flush=True)
    result = subprocess.run([sys.executable, os.path.join(HERE, "train.py"),
                             "--emb", emb, "--arch", arch])
    if result.returncode != 0:
        print(f"!! {emb}/{arch} exited {result.returncode}; continuing", flush=True)
print("queue complete", flush=True)
