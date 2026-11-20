"""Runs the trainings behind ge_contrasts.py: 22 configurations of ge_train.py
--no-cv, three trainings each, most important first. They cover masking of
the technician, customer and repair-action fields, BiLSTM against mean
pooling on every field, the chronological and unit-grouped splits, and
BiGRU and CNN on the customer and technician fields. Results go to
results_contrasts/; finished trainings are skipped, so the queue can be
stopped and restarted. Run after ge_naming_audit.py:
python ge_queue_contrasts.py [--smoke]
"""
import os
import subprocess
import sys

from ge_lexicon import HERE, MASK_FULL, MASK_NAMES, OUT

# split, field, embedding, architecture, mask file
JOBS = [
    ("random", "technician", "avi2vec", "bilstm", None),
    ("random", "technician", "avi2vec", "bilstm", MASK_FULL),
    ("random", "technician", "avi2vec", "bilstm", MASK_NAMES),
    ("random", "customer", "avi2vec", "bilstm", None),
    ("random", "customer", "avi2vec", "bilstm", MASK_FULL),
    ("random", "customer", "glove200", "bilstm", None),
    ("random", "customer", "glove200", "meanmlp", None),
    ("random", "technician", "glove200", "bilstm", None),
    ("random", "technician", "glove200", "meanmlp", None),
    ("temporal", "customer", "avi2vec", "bilstm", None),
    ("temporal", "technician", "avi2vec", "bilstm", None),
    ("random", "repair", "avi2vec", "bilstm", None),
    ("random", "repair", "avi2vec", "bilstm", MASK_FULL),
    ("unit_grouped", "customer", "avi2vec", "bilstm", None),
    ("unit_grouped", "technician", "avi2vec", "bilstm", None),
    ("random", "repair", "glove200", "bilstm", None),
    ("random", "repair", "glove200", "meanmlp", None),
    ("random", "technician", "glove200", "bilstm", MASK_FULL),
    ("random", "customer", "glove200", "bigru", None),
    ("random", "customer", "glove200", "cnn", None),
    ("random", "technician", "glove200", "bigru", None),
    ("random", "technician", "glove200", "cnn", None),
]

smoke = "--smoke" in sys.argv
env = dict(os.environ, GE_RES=OUT)
for n, (split, field, emb, arch, mask) in enumerate(JOBS, 1):
    cmd = [sys.executable, os.path.join(HERE, "ge_train.py"), "--split", split,
           "--field", field, "--emb", emb, "--arch", arch, "--no-cv"]
    if mask:
        cmd += ["--mask-file", mask]
    if smoke:
        cmd.append("--smoke")
    print(f"[{n}/{len(JOBS)}] {split} {field} {emb} {arch} {os.path.basename(mask or '')}", flush=True)
    if subprocess.run(cmd, env=env).returncode != 0:
        print("failed, continuing", flush=True)
print("queue complete", flush=True)
