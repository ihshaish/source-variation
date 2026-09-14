# MaintNet external test: registered note

Written before downloading any data; the git history carries the
timestamp. Question: does the source and target alignment observed on the
GE records recur on a public maintenance corpus? MaintNet's aviation
logbooks carry a problem narrative and an action narrative for the same
maintenance event.

## Design

Six cells: three inputs (problem text, action text, both concatenated) by
two target families (one problem-side label, one action-side label, chosen
from the annotated release by the support criteria below). Headline
contrast: problem text against action text on the action-side target, the
cross-stage cell that parallels the GE customer field predicting the repair
outcome. Diagonal cells (a field predicting a label annotated from that
field) are reported but discounted. Control: target-span masking at
evaluation; the result is the advantage that survives the mask.

Configurations: TF-IDF with logistic regression; BiLSTM over GloVe-200;
BiLSTM over word2vec trained on the task partition. No other models.

Split: stratified 80/20 held-out with near-duplicate confinement (MinHash
over 5-token shingles, Jaccard 0.8). Statistics: record-level bootstrap
intervals, paired approximate-randomisation tests, Holm within families,
three seeds. Same machinery as the ASRS pipeline.

## Go or no-go criteria

- G1: provenance of the annotated release verifiable; licence permits use.
- G2: one problem-side and one action-side target, each with at least 4
  classes of at least 150 records after minimum-support filtering and
  near-duplicate collapse.
- G3: the action-side target's span identifiable, and so maskable, in at
  least 80% of records.
- G4: at least 3000 effective events after near-duplicate collapse.
- Any failure: no-go. The paper is submitted as it stands and no substitute
  dataset is sought.

## Outcome commitments

All six cells are reported. No outcome triggers further experiments.

## Datasets considered and excluded

FAA SDR: a single narrative. NTSB: factual and cause narratives are
finalised in one report by one body, and the domain is accident
investigation. Open Repair Alliance: domain drift. MIMIC: domain drift and
access agreements. Further models, splits or tests on the GE data: none. A
blinded re-audit belongs to the follow-up study the paper specifies.

## Phase 0 outcome (recorded 2026-08-23): no-go

The annotated release (Zenodo 20779601, CC BY 4.0, 6169 records) was
inspected against the criteria.

- ACTION_TYPE is empty and PROBLEM_TYPE is 97% one value, so both released
  taxonomies are unusable.
- A verb-class taxonomy derived from the TAGGEDACTION spans passes the
  action-side support bar: replace-install 4221, inspect-check 434,
  repair-secure 418, adjust 301 (69% majority class).
- The problem side fails G2. LOCATION collapses to cylinder (3999) and
  powerplant (1028) with nothing else above 9 records; PROBLEM_PART
  collapses the same way (gaskets, covers, baffles). No problem-side
  taxonomy with four supported classes exists.
- G4 is doubtful: 3531 unique problem texts, roughly 2800 groups under an
  approximate near-duplicate collapse, fewer under the full procedure.

No-go. The one surviving comparison (the action verb class from problem
against action text) was declined: a 69% majority class and a
self-annotated target.

# Addendum (registered before tabulation): matched records inside the corpora already held

Survey criteria: several records per event; stated target provenance;
public; scale after de-duplication; records not derived from one another;
a distinct contribution to the study. Two parts result, and no more than
two.

## Part A: ASRS matched records

A1, narrative against synopsis. Same events, labels and split as the
Aircraft task. The narrative is written by the reporter before analyst
coding; the synopsis by the analyst who codes. Configurations: TF-IDF with
logistic regression; BiLSTM over GloVe-200; BiLSTM over word2vec trained on
the training partition of the respective record. Three seeds; paired
held-out tests as in the paper.
Go or no-go: median 3-shingle containment of the synopsis in its own
narrative at most 0.8 (above that the synopsis is treated as an extract and
the part is dropped); median synopsis length at least 15 tokens.

A2, reporter 1 against reporter 2. Dual-report events only: one model,
trained on primary narratives as in the paper, evaluated on each
dual-report held-out record once per narrative. Event, label, stage and
model are fixed; only the account read varies. The training domain favours
reporter 1 and the subset is multi-crew by construction; both are stated,
neither corrected for.
Go or no-go: at least 1500 dual-report records inside the task with both
narratives of usable length, at least 400 of them held out.

## Part B: NHTSA recall campaigns

Three fields per campaign (defect description, consequence, corrective
action); target: the component category at its top level. Campaign-level
de-duplication; split grouped by campaign; TF-IDF and a BiLSTM over
task-trained word2vec; three seeds.
Go or no-go: the flat file carries the three text fields; at least 8
component classes with 300 or more campaigns after de-duplication; median
field lengths at least 15 tokens each. If B fails, the Eclipse defect
dataset may be assessed once under the same criteria; if that fails too,
part B is dropped.

## Outcome commitments

All cells are reported as measured. No further datasets beyond these two
parts.
