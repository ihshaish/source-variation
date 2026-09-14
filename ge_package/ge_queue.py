"""Runs the standard training matrix by calling ge_train.py once per
configuration, the main runs first. Result keys are idempotent, so the
queue can be stopped and restarted without repeating finished runs; the
lists below can be edited. Prints "queue complete" at the end.
Run: python ge_queue.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

CORE = [  # split, field, emb, arch
    ("random", "repair", "avi2vec", "bilstm"), ("random", "repair", "glove200", "bilstm"),
    ("random", "customer", "avi2vec", "bilstm"), ("random", "customer", "glove200", "bilstm"),
    ("random", "technician", "avi2vec", "bilstm"), ("random", "technician", "glove200", "bilstm"),
    ("random", "sequential", "avi2vec", "bilstm"),
    ("random", "repair", "fasttext_ge", "bilstm"), ("random", "repair", "w2v_ge", "bilstm"),
    ("random", "repair", "avi2vec", "bigru"), ("random", "repair", "glove200", "bigru"),
    ("random", "repair", "glove200", "cnn"), ("random", "repair", "glove200", "meanmlp"),
    # grouped and temporal splits on the headline configuration
    ("dup_exact", "repair", "avi2vec", "bilstm"), ("dup_exact", "repair", "glove200", "bilstm"),
    ("dup_near", "repair", "avi2vec", "bilstm"), ("dup_near", "repair", "glove200", "bilstm"),
    ("temporal", "repair", "avi2vec", "bilstm"), ("temporal", "repair", "glove200", "bilstm"),
    ("unit_grouped", "repair", "avi2vec", "bilstm"), ("unit_grouped", "repair", "glove200", "bilstm"),
]
MASKED = [  # mask files come from ge_strata.py --make-mask and ge_leakage.py
    ("random", "repair", "avi2vec", "bilstm", "masks/vdelta.txt"),
    ("random", "repair", "glove200", "bilstm", "masks/vdelta.txt"),
    ("random", "repair", "avi2vec", "bilstm", "masks/outcome_terms.txt"),
    ("random", "repair", "glove200", "bilstm", "masks/outcome_terms.txt"),
]
for config in CORE:
    split, field, emb, arch = config
    args = [sys.executable, os.path.join(HERE, "ge_train.py"), "--split", split,
            "--field", field, "--emb", emb, "--arch", arch]
    print("===", *config, flush=True)
    if subprocess.run(args).returncode != 0:
        print("!! failed, continuing", flush=True)
for split, field, emb, arch, mask in MASKED:
    if not os.path.exists(os.path.join(HERE, mask)):
        print(f"skip masked run ({mask} missing - run ge_strata.py --make-mask / ge_leakage.py first)")
        continue
    args = [sys.executable, os.path.join(HERE, "ge_train.py"), "--split", split,
            "--field", field, "--emb", emb, "--arch", arch, "--mask-file",
            os.path.join(HERE, mask)]
    print("=== masked:", split, field, emb, arch, mask, flush=True)
    if subprocess.run(args).returncode != 0:
        print("!! failed, continuing", flush=True)
print("queue complete", flush=True)
