"""Runs train.py for the text CNN and the mean-pooling MLP with GloVe-200.

Same setup as run_queue.py, meant to run after it. The CNN is a sequence
model without recurrence; the MLP uses no word order at all.
Run: python3 run_queue_architectures.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = [("glove200", "cnn"), ("glove200", "meanmlp")]

for emb, arch in CONFIGS:
    print(f"=== {emb} / {arch} ===", flush=True)
    result = subprocess.run([sys.executable, os.path.join(HERE, "train.py"),
                             "--emb", emb, "--arch", arch])
    if result.returncode != 0:
        print(f"!! {emb}/{arch} exited {result.returncode}; continuing", flush=True)
print("queue2 complete", flush=True)
