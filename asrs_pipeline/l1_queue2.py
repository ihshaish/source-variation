"""Architecture-probe queue (runs after l1_queue.py): TextCNN and mean-pool MLP
under the reference embedding, same setup. Answers the "BiLSTM vs BiGRU is a
narrow architecture contrast" critique: CNN = non-recurrent sequence model,
MeanMLP = no sequence modelling at all. GE equivalents go to Peter (plan P2b).
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = [("glove200", "cnn"), ("glove200", "meanmlp")]

for emb, arch in CONFIGS:
    print(f"=== {emb} / {arch} ===", flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, "l1_train.py"),
                        "--emb", emb, "--arch", arch])
    if r.returncode != 0:
        print(f"!! {emb}/{arch} exited {r.returncode}; continuing", flush=True)
print("queue2 complete", flush=True)
