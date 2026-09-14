"""Prints the near-duplicate split results as two tables and checks the
criterion fixed before the run: every BiLSTM summary contrast positive and
Holm-significant, and both TF-IDF summary contrasts positive. Reads the JSON
files under results/nhtsa/neardup/ and the reference-split results for
comparison; writes nothing.

    python3 tabulate_neardup.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results", "nhtsa")
NEARDUP = os.path.join(RESULTS, "neardup")

grouping = json.load(open(os.path.join(NEARDUP, "nhtsa_neardup_grouping.json")))
scores = {row["key"]: row["f1"] for row in json.load(open(os.path.join(NEARDUP, "nhtsa_neardup_results.json")))}
contrasts = json.load(open(os.path.join(NEARDUP, "nhtsa_neardup_contrasts.json")))
reference = {row["key"]: row["f1"] for row in json.load(open(os.path.join(RESULTS, "nhtsa_task_results.json")))}

FIELDS = {"summary": "Defect summary", "conseq": "Consequence", "remedy": "Remedy"}


def contrast(model, training, a, b):
    for row in contrasts:
        if row["contrast"] == f"{model} s{training}: {a} vs {b}":
            return row


bilstm = [contrast("bilstm", s, "summary", b) for s in (0, 1, 2) for b in ("conseq", "remedy")]
tfidf = [contrast("tfidf", 0, "summary", b) for b in ("conseq", "remedy")]
criterion_met = all(row["delta"] > 0 and row["p_holm"] < 0.05 for row in bilstm) and all(row["delta"] > 0 for row in tfidf)

n_task = grouping["n_task"]
n_test = grouping["n_test"]
n_test_reference = grouping["n_test_leg2"]
share = 100 * grouping["campaigns_in_multi_groups"] / n_task

print("Grouping")
print(f"  exact-key groups {grouping['exact_groups_leg2']}, near-duplicate groups {grouping['groups_neardup']}")
print(f"  campaigns in groups with more than one member: {grouping['campaigns_in_multi_groups']} of {n_task} ({share:.1f}%)")
print(f"  largest group: {grouping['largest_group']} campaigns")
print(f"  train/test {n_task - n_test}/{n_test}; reference split {n_task - n_test_reference}/{n_test_reference}")
print("  The held-out campaigns differ from the reference split, so compare orderings and not absolute scores.")
print()
print("Held-out macro-F1 (reference split in brackets)")
print("| Field | TF-IDF | BiLSTM 1 | BiLSTM 2 | BiLSTM 3 |")
print("|---|---|---|---|---|")
for key, name in FIELDS.items():
    cells = [f"{scores[f'nhtsa_{key}_tfidf_s0']:.3f} ({reference[f'nhtsa_{key}_tfidf_s0']:.3f})"]
    for s in (0, 1, 2):
        cells.append(f"{scores[f'nhtsa_{key}_bilstm_s{s}']:.3f} ({reference[f'nhtsa_{key}_bilstm_s{s}']:.3f})")
    print(f"| {name} | " + " | ".join(cells) + " |")
print()
print("Contrasts (difference in macro-F1, Holm p within the twelve-contrast family)")
print("| Contrast | TF-IDF | BiLSTM 1 | BiLSTM 2 | BiLSTM 3 |")
print("|---|---|---|---|---|")
for a, b, name in (("summary", "conseq", "Summary minus consequence"),
                   ("summary", "remedy", "Summary minus remedy"),
                   ("remedy", "conseq", "Remedy minus consequence")):
    row = contrast("tfidf", 0, a, b)
    cells = [f"{row['delta']:+.3f} ({row['p_holm']:.4f})"]
    for s in (0, 1, 2):
        row = contrast("bilstm", s, a, b)
        cells.append(f"{row['delta']:+.3f} ({row['p_holm']:.4f})")
    print(f"| {name} | " + " | ".join(cells) + " |")
print()
print("Criterion fixed before the run:", "met" if criterion_met else "not met")
