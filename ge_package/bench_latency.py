"""Times CPU inference for the four architectures of ge_lib.make_model at the
three field length caps, over a frozen 200-dimensional embedding table. For
each pair it reports the trainable and total parameter counts, the median and
95th-percentile latency of a single-record forward pass, and the throughput at
batch 128. Inputs are random token ids filling the whole length cap, and
since the models do not skip padding a shorter record costs the same.
Tokenising and encoding the text are not timed. Prints a table and writes
latency.json next to this script, or to the path given.
Run: python bench_latency.py [--threads 1] [--out latency.json]
"""
import argparse
import json
import os
import platform
import time

import numpy as np
import torch

from ge_lib import FIELDS, MAX_LEN, N_RESERVED, make_model

VOCAB_ROWS = 20_000
EMB_DIM = 200
WARM_UP = 20
TIMED_SINGLE = 200
TIMED_BATCH = 50
BATCH = 128
ARCHS = ("meanmlp", "bilstm", "bigru", "cnn")


def time_forwards(model, x, repeats):
    """Wall-clock seconds of each of `repeats` forward passes, after warm-up."""
    with torch.no_grad():
        for _ in range(WARM_UP):
            model(x)
        seconds = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(x)
            seconds.append(time.perf_counter() - start)
    return np.array(seconds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "latency.json"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    matrix = rng.normal(0, 0.4, (VOCAB_ROWS, EMB_DIM)).astype(np.float32)

    report = {"machine": platform.machine(), "processor": platform.processor(),
              "system": platform.platform(), "python": platform.python_version(),
              "torch": torch.__version__, "numpy": np.__version__,
              "threads": torch.get_num_threads(), "vocab_rows": VOCAB_ROWS,
              "embedding_dim": EMB_DIM, "timed_single": TIMED_SINGLE,
              "timed_batch": TIMED_BATCH, "batch": BATCH, "rows": []}
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}))
    print(f"{'field':10s} {'cap':>4s} {'model':8s} {'trainable':>9s} {'total':>9s} "
          f"{'median ms':>9s} {'p95 ms':>7s} {'records/s':>9s}")
    for field in FIELDS:
        cap = MAX_LEN[field]
        single = torch.from_numpy(rng.integers(N_RESERVED, VOCAB_ROWS, (1, cap)))
        batch = torch.from_numpy(rng.integers(N_RESERVED, VOCAB_ROWS, (BATCH, cap)))
        for arch in ARCHS:
            model = make_model(matrix, arch)
            model.eval()
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            single_ms = 1000 * time_forwards(model, single, TIMED_SINGLE)
            batch_s = time_forwards(model, batch, TIMED_BATCH)
            row = {"field": field, "max_len": cap, "arch": arch,
                   "trainable_parameters": trainable, "total_parameters": total,
                   "latency_ms_median": round(float(np.median(single_ms)), 3),
                   "latency_ms_p95": round(float(np.percentile(single_ms, 95)), 3),
                   "records_per_second_batch128": round(BATCH / float(np.median(batch_s)))}
            report["rows"].append(row)
            print(f"{field:10s} {cap:4d} {arch:8s} {trainable:9d} {total:9d} "
                  f"{row['latency_ms_median']:9.3f} {row['latency_ms_p95']:7.3f} "
                  f"{row['records_per_second_batch128']:9d}")
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
