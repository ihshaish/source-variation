"""L1/L2 run queue — sequential (single MPS device), resumable.

Order puts the headline thesis comparison (GloVe-200, both architectures) and
the new science (fastText, L2) first, remaining GloVe dimensionalities after.
Avi2Vec is appended automatically if the cleared vectors appear as
l1_data/avi2vec.kv (see PAPER_A_EXPERIMENT_PLAN.md §2).
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
if os.path.exists(os.path.join(HERE, "l1_data", "avi2vec.kv")):
    CONFIGS += [("avi2vec", "bilstm"), ("avi2vec", "bigru")]

for emb, arch in CONFIGS:
    print(f"=== {emb} / {arch} ===", flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, "l1_train.py"),
                        "--emb", emb, "--arch", arch])
    if r.returncode != 0:
        print(f"!! {emb}/{arch} exited {r.returncode}; continuing", flush=True)
print("queue complete", flush=True)
