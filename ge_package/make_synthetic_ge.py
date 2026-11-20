"""Writes invented repair records in the input schema of ge_build.py, so that
the scripts run without the proprietary data. The file has 5,792 events with
the published class counts, and its field lengths and assembly naming rates
per class follow the published aggregates. All wording, assembly names,
dates, unit serials and operators are made up, and random 200-dimensional
vectors for the invented words stand in for the GloVe file. Writes
ge_records.csv under GE_DATA and glove.6B.200d.txt under EMB_DIR.
Run: python make_synthetic_ge.py
"""
import csv
import datetime
import os

import numpy as np

from ge_lib import DATA, tokenize

SEED = 20260802
EMB_DIR = os.environ.get("EMB_DIR", os.path.join(DATA, "..", "emb"))
# processor only, key panel only, both, neither
CLASS_COUNTS = (2488, 386, 2010, 908)
MEAN_CHARS = {"customer": 50, "technician": 108, "repair": 215}
# length targets handed to the text builder, set so that the realised mean
# lengths come out at MEAN_CHARS (a text ends on the phrase that passes its target)
TARGET_CHARS = {"customer": 42, "technician": 95, "repair": 203}
# invented names for the two assemblies, the processor's stand-in first
ASSEMBLIES = ("controller", "keypad")

# share of events whose customer text names any assembly, by class
CUSTOMER_NAMES = (0.160, 0.166, 0.183, 0.120)
# technician text, single-assembly classes: (replaced assembly named, other assembly named)
TECHNICIAN_NAMES = {0: (0.508, 0.147), 1: (0.738, 0.176)}
# technician text, both replaced: both named, and exactly one named
TECHNICIAN_BOTH = 0.538
TECHNICIAN_ONE = 0.823 - 0.538
# technician text, nothing replaced: some assembly named
TECHNICIAN_NEITHER = 0.219
# share of events whose repair text names an assembly, by class
REPAIR_NAMES = (0.996, 0.995, 0.999, 0.324)

CUSTOMER_PHRASES = [
    "screen dark on start", "unit resets in use", "buttons stick", "no response to input",
    "backlight dim", "fault light on", "freezes when warm", "intermittent", "lines across screen",
    "will not power up", "slow to respond", "screen flickers", "reported by crew",
    "removed for check", "fault after take-off", "input ignored", "goes blank then recovers",
]
CUSTOMER_NAMING = ["see advisory on {}", "suspect {}", "advisory note {} applies"]
TECHNICIAN_PHRASES = [
    "fault confirmed on bench", "unit fails start-up test", "no output on second channel",
    "visual check satisfactory", "connector pins checked and clean", "reset seen after warm soak",
    "backlight uneven across the screen", "several buttons give no signal",
    "could not repeat the reported fault", "cover removed for inspection",
    "supply rails within limits", "self test stops at stage three", "moisture marks inside case",
    "screen image breaks up when tapped", "memory check reports errors", "runs normally when cold",
]
TECHNICIAN_NAMING = ["fault traced to {}", "{} suspect", "signal lost at {}", "{} shows wear",
                     "errors logged against {}"]
REPAIR_PHRASES = [
    "unit cleaned", "software reloaded to current standard", "full functional test passed",
    "soak test completed with no faults", "case seals checked", "connectors cleaned and reseated",
    "backlight level adjusted", "calibration carried out", "final inspection completed",
    "unit returned to service", "fasteners torqued and sealed", "test record attached",
    "burn-in run for four hours", "labels renewed", "paperwork raised for release",
    "earth bond measured", "screen surface polished", "vent filter cleared",
    "internal wiring secured", "power-up cycles repeated ten times", "packed for dispatch",
    "front frame straightened", "pressure test satisfactory", "old sealant removed",
]
REPAIR_REPLACING = ["replaced {}", "new {} fitted", "{} replaced and tested", "{} changed"]
REPAIR_KEEPING = ["{} reseated", "{} cleaned and tested", "{} inspected, no defect"]
REPAIR_NO_FAULT = ["no fault found", "tested to specification, no fault found"]


def build_text(rng, required, pool, target):
    """Starts from the required phrases and adds phrases from the pool, each at
    most once, until the text reaches the target length. The phrase order is
    shuffled."""
    phrases = list(required)
    length = sum(len(phrase) + 2 for phrase in phrases)
    for i in rng.permutation(len(pool)):
        if length >= target:
            break
        phrases.append(pool[i])
        length += len(pool[i]) + 2
    order = rng.permutation(len(phrases))
    return ", ".join(phrases[i] for i in order)


def pick(rng, templates, assembly):
    return templates[rng.integers(len(templates))].format(assembly)


def named_by_technician(rng, label):
    """Which assemblies the technician text names, as a list."""
    if label in TECHNICIAN_NAMES:
        replaced_rate, other_rate = TECHNICIAN_NAMES[label]
        replaced = ASSEMBLIES[label]
        other = ASSEMBLIES[1 - label]
        named = []
        if rng.random() < replaced_rate:
            named.append(replaced)
        if rng.random() < other_rate:
            named.append(other)
        return named
    draw = rng.random()
    if label == 2:
        if draw < TECHNICIAN_BOTH:
            return list(ASSEMBLIES)
        if draw < TECHNICIAN_BOTH + TECHNICIAN_ONE:
            return [ASSEMBLIES[rng.integers(2)]]
        return []
    if draw < TECHNICIAN_NEITHER:
        return [ASSEMBLIES[rng.integers(2)]]
    return []


def make_record(rng, label):
    target = {}
    for field, chars in TARGET_CHARS.items():
        target[field] = rng.gamma(4.0, chars / 4.0)

    required = []
    if rng.random() < CUSTOMER_NAMES[label]:
        required.append(pick(rng, CUSTOMER_NAMING, ASSEMBLIES[rng.integers(2)]))
    customer = build_text(rng, required, CUSTOMER_PHRASES, target["customer"])

    required = []
    for assembly in named_by_technician(rng, label):
        required.append(pick(rng, TECHNICIAN_NAMING, assembly))
    technician = build_text(rng, required, TECHNICIAN_PHRASES, target["technician"])

    required = []
    if label == 3:
        required.append(REPAIR_NO_FAULT[rng.integers(len(REPAIR_NO_FAULT))])
        if rng.random() < REPAIR_NAMES[3]:
            required.append(pick(rng, REPAIR_KEEPING, ASSEMBLIES[rng.integers(2)]))
    else:
        replaced = list(ASSEMBLIES) if label == 2 else [ASSEMBLIES[label]]
        for assembly in replaced:
            if rng.random() < REPAIR_NAMES[label]:
                required.append(pick(rng, REPAIR_REPLACING, assembly))
            else:
                required.append("faulty assembly replaced")
    repair = build_text(rng, required, REPAIR_PHRASES, target["repair"])
    return customer, technician, repair


def named_share(rows, field, label, assemblies, need_all=False):
    """Percentage of the class whose field names any, or all, of the assemblies."""
    hits = []
    for row in rows:
        if row["label"] == label:
            tokens = set(tokenize(row[field]))
            found = [assembly in tokens for assembly in assemblies]
            hits.append(all(found) if need_all else any(found))
    return 100 * np.mean(hits)


def main():
    rng = np.random.default_rng(SEED)
    labels = np.repeat(np.arange(4), CLASS_COUNTS)
    rng.shuffle(labels)
    first_day = datetime.date(2017, 1, 1)
    rows = []
    for i, label in enumerate(labels):
        customer, technician, repair = make_record(rng, int(label))
        day = first_day + datetime.timedelta(days=int(rng.integers(3650)))
        unit = f"U{rng.integers(4200):04d}" if rng.random() > 0.01 else ""
        operator = f"OP{rng.integers(30):02d}" if rng.random() > 0.05 else ""
        rows.append({"record_id": f"S{i + 1:05d}", "date": day.isoformat(),
                     "customer": customer, "technician": technician, "repair": repair,
                     "label": int(label), "unit_serial": unit, "operator": operator})

    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, "ge_records.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    vocab = set()
    for row in rows:
        for field in MEAN_CHARS:
            vocab.update(tokenize(row[field]))
    os.makedirs(EMB_DIR, exist_ok=True)
    with open(os.path.join(EMB_DIR, "glove.6B.200d.txt"), "w", encoding="utf-8") as f:
        for word in sorted(vocab):
            vector = rng.normal(0, 0.4, 200)
            f.write(word + " " + " ".join(f"{value:.4f}" for value in vector) + "\n")

    print(f"{len(rows)} records, {len(vocab)} vocabulary words")
    print("class counts:", np.bincount(labels).tolist())
    for field, published in MEAN_CHARS.items():
        mean_chars = np.mean([len(row[field]) for row in rows])
        shares = [named_share(rows, field, label, ASSEMBLIES) for label in range(4)]
        print(f"{field:10s} mean length {mean_chars:5.1f} characters (published {published}); "
              "names an assembly, % by class: " + " / ".join(f"{share:.1f}" for share in shares))
    print("technician, % naming: the replaced assembly "
          f"{named_share(rows, 'technician', 0, ASSEMBLIES[:1]):.1f} (class 0) "
          f"{named_share(rows, 'technician', 1, ASSEMBLIES[1:]):.1f} (class 1); "
          f"the other assembly {named_share(rows, 'technician', 0, ASSEMBLIES[1:]):.1f} "
          f"{named_share(rows, 'technician', 1, ASSEMBLIES[:1]):.1f}; "
          f"both, class 2: {named_share(rows, 'technician', 2, ASSEMBLIES, need_all=True):.1f}")


if __name__ == "__main__":
    main()
