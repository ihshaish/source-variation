"""Runs the whole GE package on invented data. No GE record ever touches this.

Generates a fake CSV and tiny GloVe stubs, then walks build -> embeds ->
train (smoke) -> strata mask -> leakage -> baselines -> stats. If this passes
on your machine, the pipeline logic is fine; the numbers are meaningless by
design. python ge_selftest.py
"""
import csv, json, os, random, subprocess, sys, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
TD = os.path.join(HERE, "_selftest")
os.environ["GE_DATA"] = os.path.join(TD, "ge_data")
os.environ["GE_RES"] = os.path.join(TD, "results")
os.environ["EMB_DIR"] = os.path.join(TD, "emb")
shutil.rmtree(TD, ignore_errors=True)
os.makedirs(os.environ["GE_DATA"]); os.makedirs(os.environ["EMB_DIR"])

random.seed(0)
words = [f"w{i}" for i in range(300)]
CLS = {0: "processor", 1: "keypanel", 2: "nff", 3: "display"}
rows = []
for i in range(400):
    c = random.randrange(4)
    filler = lambda n: " ".join(random.choices(words, k=n))
    rows.append({"record_id": f"r{i}", "date": f"20{17+random.randrange(8)}0{1+random.randrange(9)}",
                 "customer": filler(10),
                 "technician": filler(20) + " " + CLS[c],
                 "repair": f"replaced {CLS[c]} " + filler(15),
                 "label": str(c), "unit_serial": f"u{i%40}", "operator": f"op{i%7}"})
with open(os.path.join(os.environ["GE_DATA"], "ge_records.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
for name, dim in [("glove.6B.200d.txt", 200), ("glove.6B.50d.txt", 50)]:
    with open(os.path.join(os.environ["EMB_DIR"], name), "w") as f:
        for wd in words[:200] + list(CLS.values()) + ["replaced"]:
            f.write(wd + " " + " ".join(f"{random.uniform(-1,1):.3f}" for _ in range(dim)) + "\n")

def run(*args, must=True):
    r = subprocess.run([sys.executable] + list(args), cwd=HERE)
    if must and r.returncode != 0:
        print(f"SELFTEST FAIL at: {args}"); sys.exit(1)

run("ge_build.py")
run("ge_embeds.py")
run("ge_train.py", "--field", "repair", "--emb", "glove200", "--arch", "bilstm", "--smoke")
run("ge_train.py", "--field", "sequential", "--emb", "fasttext_ge", "--arch", "cnn", "--smoke")
run("ge_train.py", "--field", "repair", "--emb", "glove200", "--arch", "bilstm",
    "--split", "dup_exact", "--smoke")
# avi2vec stand-in: reuse w2v_ge as the .kv for the strata path
from gensim.models import Word2Vec
Word2Vec.load(os.path.join(os.environ["GE_DATA"], "w2v_ge_200d.bin")).wv.save(
    os.path.join(os.environ["GE_DATA"], "avi2vec.kv"))
run("ge_strata.py", "--make-mask")
run("ge_train.py", "--field", "repair", "--emb", "glove200", "--arch", "bilstm",
    "--mask-file", os.path.join(HERE, "masks", "vdelta.txt"), "--smoke")
run("ge_leakage.py")
run("ge_baselines.py")
print("\nSELFTEST PASSED - package is runnable end-to-end.")
shutil.rmtree(TD, ignore_errors=True)
