"""Runs the whole package on invented data; no GE record is involved.
Writes a fake records CSV and small GloVe stand-ins into a _selftest folder
next to this script, points GE_DATA, GE_RES and EMB_DIR at it, and then runs
the build, the embeddings, training in smoke mode, the mask, the leakage
audit, the baselines, the naming audit, the masked TF-IDF baselines, the
contrast queue and the contrast statistics, the last on noisy stand-in
predictions. Prints SELFTEST PASSED when every step exits cleanly; the
numbers are not meaningful. Run: python ge_selftest.py
"""
import csv
import json
import os
import random
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(HERE, "_selftest")
os.environ["GE_DATA"] = os.path.join(TEST_DIR, "ge_data")
os.environ["GE_RES"] = os.path.join(TEST_DIR, "results")
os.environ["EMB_DIR"] = os.path.join(TEST_DIR, "emb")
os.environ["GE_RES_CONTRASTS"] = os.path.join(TEST_DIR, "results_contrasts")
os.environ["GE_PRIVATE"] = os.path.join(TEST_DIR, "private")
os.environ["GE_MASKS"] = os.path.join(TEST_DIR, "masks")
shutil.rmtree(TEST_DIR, ignore_errors=True)
os.makedirs(os.environ["GE_DATA"])
os.makedirs(os.environ["EMB_DIR"])

random.seed(0)
words = [f"w{i}" for i in range(300)]
CLASS_NAMES = {0: "processor", 1: "keypanel", 2: "nff", 3: "display"}


def filler(count):
    return " ".join(random.choices(words, k=count))


rows = []
for i in range(400):
    label = random.randrange(4)
    rows.append({"record_id": f"r{i}", "date": f"20{17 + random.randrange(8)}0{1 + random.randrange(9)}",
                 "customer": filler(10),
                 "technician": filler(20) + " " + CLASS_NAMES[label],
                 "repair": f"replaced {CLASS_NAMES[label]} " + filler(15),
                 "label": str(label), "unit_serial": f"u{i % 40}", "operator": f"op{i % 7}"})
with open(os.path.join(os.environ["GE_DATA"], "ge_records.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
for name, dim in [("glove.6B.200d.txt", 200), ("glove.6B.50d.txt", 50)]:
    with open(os.path.join(os.environ["EMB_DIR"], name), "w") as f:
        for word in words[:200] + list(CLASS_NAMES.values()) + ["replaced"]:
            f.write(word + " " + " ".join(f"{random.uniform(-1, 1):.3f}" for _ in range(dim)) + "\n")


def run(*args, must=True):
    result = subprocess.run([sys.executable] + list(args), cwd=HERE)
    if must and result.returncode != 0:
        print(f"SELFTEST FAIL at: {args}")
        sys.exit(1)


run("ge_build.py")
run("ge_embeds.py")
run("ge_train.py", "--field", "repair", "--emb", "glove200", "--arch", "bilstm", "--smoke")
run("ge_train.py", "--field", "sequential", "--emb", "fasttext_ge", "--arch", "cnn", "--smoke")
run("ge_train.py", "--field", "repair", "--emb", "glove200", "--arch", "bilstm",
    "--split", "dup_exact", "--smoke")
# the mask step needs avi2vec.kv, so the word2vec vectors just trained stand in for it
from gensim.models import Word2Vec
Word2Vec.load(os.path.join(os.environ["GE_DATA"], "w2v_ge_200d.bin")).wv.save(
    os.path.join(os.environ["GE_DATA"], "avi2vec.kv"))
run("ge_strata.py", "--make-mask")
run("ge_train.py", "--field", "repair", "--emb", "glove200", "--arch", "bilstm",
    "--mask-file", os.path.join(HERE, "masks", "vdelta.txt"), "--smoke")
run("ge_leakage.py")
run("ge_baselines.py")
run("ge_terms.py")
run("ge_naming_audit.py")
run("ge_tfidf_masked.py")
run("ge_queue_contrasts.py", "--smoke")

# Smoke runs train once on a subsample, so ge_contrasts.py gets noisy
# full-length predictions for three trainings of every queued configuration.
from ge_lib import load_records
records = load_records()
splits = json.load(open(os.path.join(os.environ["GE_DATA"], "splits.json")))
rng = np.random.default_rng(0)
for name in os.listdir(os.environ["GE_RES_CONTRASTS"]):
    if not (name.startswith("preds_smoke_") and name.endswith("_s0.npz")):
        continue
    key = name[len("preds_smoke_"):-len("_s0.npz")]
    held_out = set(splits[key.split("_technician")[0].split("_customer")[0].split("_repair")[0]])
    y = np.array([r["label"] for r in records if r["id"] in held_out])
    for folder in (os.environ["GE_RES"], os.environ["GE_RES_CONTRASTS"]):
        for seed in (0, 1, 2):
            noisy = np.where(rng.random(len(y)) < 0.6, y, rng.integers(0, 4, len(y)))
            np.savez(os.path.join(folder, f"preds_{key}_s{seed}.npz"), yhat=noisy, y=y)
run("ge_contrasts.py", "--quick")
print("\nSELFTEST PASSED - package is runnable end-to-end.")
shutil.rmtree(TEST_DIR, ignore_errors=True)
