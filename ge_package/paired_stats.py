"""Paired statistics on the held-out predictions of the reference partition.
For each contrast A minus B the estimate is the mean over the three trainings
of the macro-F1 difference on the same held-out events. The interval is a
paired bootstrap over events, with one set of resampled indices shared by both
conditions and all trainings, and is repeated with units resampled in place of
events. The p-value is a two-sided approximate randomisation test with one
swap per event shared across trainings, Holm-adjusted within each family.
Reads preds_random_*.npz from GE_RES and the record file and splits from
GE_DATA, and writes paired_stats.json to GE_RES. Run: python paired_stats.py
"""
import json
import os

import numpy as np

from ge_lib import DATA, RES, load_records, macro_f1

N_RESAMPLES = 10_000
SEEDS = (0, 1, 2)
# family, description, condition A, condition B
CONTRASTS = [
    ("record", "technician minus customer, BiLSTM",
     "technician_glove200_bilstm", "customer_glove200_bilstm"),
    ("model", "customer: BiLSTM minus mean pooling",
     "customer_glove200_bilstm", "customer_glove200_meanmlp"),
    ("model", "technician: BiLSTM minus mean pooling",
     "technician_glove200_bilstm", "technician_glove200_meanmlp"),
    ("model", "repair action: BiLSTM minus mean pooling",
     "repair_glove200_bilstm", "repair_glove200_meanmlp"),
]


def load_runs(key):
    """Held-out labels and the predictions of the three trainings."""
    runs = []
    for seed in SEEDS:
        data = np.load(os.path.join(RES, f"preds_random_{key}_s{seed}.npz"))
        runs.append(data["yhat"])
    return data["y"], runs


def mean_difference(y, runs_a, runs_b):
    differences = []
    for yhat_a, yhat_b in zip(runs_a, runs_b):
        differences.append(macro_f1(y, yhat_a) - macro_f1(y, yhat_b))
    return float(np.mean(differences))


def bootstrap_ci(y, runs_a, runs_b, groups, rng):
    """Percentile interval. Whole groups are resampled, and a group is one
    event when groups is None."""
    if groups is None:
        members = [np.array([i]) for i in range(len(y))]
    else:
        members = [np.where(groups == group)[0] for group in np.unique(groups)]
    draws = np.empty(N_RESAMPLES)
    for b in range(N_RESAMPLES):
        chosen = rng.integers(0, len(members), len(members))
        sample = np.concatenate([members[i] for i in chosen])
        resampled_a = [yhat[sample] for yhat in runs_a]
        resampled_b = [yhat[sample] for yhat in runs_b]
        draws[b] = mean_difference(y[sample], resampled_a, resampled_b)
    return [round(float(np.percentile(draws, 2.5)), 4), round(float(np.percentile(draws, 97.5)), 4)]


def randomisation_p(y, runs_a, runs_b, rng):
    observed = abs(mean_difference(y, runs_a, runs_b))
    count = 0
    for _ in range(N_RESAMPLES):
        swap = rng.random(len(y)) < 0.5
        swapped_a = [np.where(swap, yhat_b, yhat_a) for yhat_a, yhat_b in zip(runs_a, runs_b)]
        swapped_b = [np.where(swap, yhat_a, yhat_b) for yhat_a, yhat_b in zip(runs_a, runs_b)]
        if abs(mean_difference(y, swapped_a, swapped_b)) >= observed - 1e-12:
            count += 1
    return (count + 1) / (N_RESAMPLES + 1)


def holm(p_values):
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def main():
    test_ids = set(json.load(open(os.path.join(DATA, "splits.json")))["random"])
    held_out = [record for record in load_records() if record["id"] in test_ids]
    units = np.array([record["unit"] or record["id"] for record in held_out])
    labels = np.array([record["label"] for record in held_out])

    rng = np.random.default_rng(20261115)
    rows = []
    for family, description, key_a, key_b in CONTRASTS:
        y, runs_a = load_runs(key_a)
        _, runs_b = load_runs(key_b)
        if not np.array_equal(y, labels):
            raise SystemExit(f"{key_a}: predictions are not in held-out file order")
        rows.append({"family": family, "contrast": description, "n_events": len(y),
                     "estimate": round(mean_difference(y, runs_a, runs_b), 4),
                     "ci95": bootstrap_ci(y, runs_a, runs_b, None, rng),
                     "ci95_unit_cluster": bootstrap_ci(y, runs_a, runs_b, units, rng),
                     "p": randomisation_p(y, runs_a, runs_b, rng)})

    for family in sorted({row["family"] for row in rows}):
        members = [row for row in rows if row["family"] == family]
        adjusted = holm([row["p"] for row in members])
        for row, p_adjusted in zip(members, adjusted):
            row["p_holm"] = round(float(p_adjusted), 4)
            row["p"] = round(row["p"], 4)

    with open(os.path.join(RES, "paired_stats.json"), "w") as f:
        json.dump({"resamples": N_RESAMPLES, "contrasts": rows}, f, indent=1)
    print(f"{'estimate':>8s}  {'95% CI':>18s}  {'unit-cluster CI':>18s}  {'p_holm':>6s}  contrast")
    for row in rows:
        ci = "[{:+.4f}, {:+.4f}]".format(*row["ci95"])
        unit_ci = "[{:+.4f}, {:+.4f}]".format(*row["ci95_unit_cluster"])
        print(f"{row['estimate']:+8.4f}  {ci:>18s}  {unit_ci:>18s}  {row['p_holm']:6.4f}  "
              f"{row['contrast']}")


if __name__ == "__main__":
    main()
