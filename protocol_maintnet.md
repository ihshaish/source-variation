# MaintNet external test: registered note

Written before downloading any data (the git history is the timestamp).
The question: does the source/target alignment observed on the GE records
recur on a public maintenance corpus? MaintNet's aviation logbooks carry a
problem narrative and an action narrative for the same maintenance event,
which is the structure the question needs.

## Design, fixed now

Six cells: three inputs (problem text, action text, both concatenated) by
two target families (one problem-side label, one action-side label, chosen
from the annotated release by the support criteria below). The headline
contrast is pre-specified as problem text vs action text on the action-side
target -- cross-stage prediction, the cell that parallels the GE customer
field predicting the repair outcome. Diagonal cells (a field predicting a
label annotated from that same field) are reported but discounted; the
mandatory control is target-span masking at evaluation, and whatever
advantage survives the mask is the result.

Three configurations, frozen: TF-IDF with logistic regression; BiLSTM over
GloVe-200; BiLSTM over word2vec trained on the task partition. No other
models, whatever the results look like.

Split: stratified 80/20 held-out with near-duplicate confinement (MinHash
over 5-token shingles, Jaccard 0.8, as in the paper). Statistics: record-
level bootstrap CIs and paired approximate-randomisation tests, Holm within
families, three seeds. Same machinery as the ASRS pipeline.

## Go / no-go, fixed now

- G1: provenance of the annotated release verifiable, licence permits use.
- G2: one problem-side and one action-side target each having at least 4
  classes with at least 150 records after minimum-support filtering and
  near-duplicate collapse.
- G3: the action-side target's span is identifiable (and hence maskable) in
  at least 80% of records.
- G4: at least 3000 effective events after near-duplicate collapse.
- Any failure: no-go. The current paper submits as is; no substitute
  dataset will be sought.

## Outcome commitments

Whatever the six cells show is what gets reported: recurrence, crossover,
or a null are all informative and none triggers further experiments.

## Closed permanently (considered, excluded, reasons)

FAA SDR: single narrative, cannot improve identification of any open claim.
NTSB: factual and cause narratives are finalised in one report by one body;
stage separation is editorial, and the domain drifts to accident
investigation. Open Repair Alliance: domain drift. MIMIC: domain drift and
access agreements. Further models, splits or tests on the GE data: the
vertical axis is saturated. Blinded re-audit: the follow-up study the paper
already specifies, not this paper.

## Phase 0 outcome (recorded 2026-10-11): no-go

The annotated release (Zenodo 20779601, CC BY 4.0, 6169 records) was
inspected against the criteria above. Findings:

- The released ACTION_TYPE column is empty and PROBLEM_TYPE is 97% one
  value, so both advertised taxonomies are unusable as shipped.
- A verb-class taxonomy derived from the TAGGEDACTION spans passes the
  action-side support bar, barely: replace-install 4221, inspect-check 434,
  repair-secure 418, adjust 301 (69% majority class).
- The problem side fails G2 outright. The fleet's logbook is topically
  homogeneous: LOCATION collapses to cylinder (3999) and powerplant (1028)
  with nothing else above 9 records, and PROBLEM_PART collapses the same
  way (the frequent parts are all gaskets, covers and baffles). No
  problem-side taxonomy with four supported classes exists, so the
  input-by-target matrix cannot be built.
- G4 is also doubtful: 3531 unique problem texts, roughly 2800 groups
  under an approximate near-duplicate collapse, and the true figure under
  the full procedure would be lower.

Per the criteria, any failure is a no-go, and the fallback applies: the
paper submits as the bounded two-corpus study it is, and no substitute
dataset will be sought. The single-target comparison that survives
(predicting the action verb class from problem versus action text) was
considered and declined: with a 69% majority class and a self-annotated
target it would weaken rather than strengthen the paper.

# Addendum (registered before tabulation): matched views inside the corpora we already hold

The MaintNet no-go closed the reviewed candidates, not the question. A wider
survey (rubric: multiple views per event; stated target provenance; public;
scale after dedup; non-derivative views; and a distinct axis contribution)
leaves exactly two experimental legs, capped at two whatever the results:

## Leg A: ASRS matched views (data already in the paper)

A1, narrative vs synopsis. Same events, same labels, same split as the
existing Aircraft task; the reporter's narrative is written before analyst
coding, the synopsis by the analyst who codes. Configurations frozen:
TF-IDF + logistic regression; BiLSTM over GloVe-200; BiLSTM over word2vec
trained on the training partition of the respective view. Three seeds,
paired held-out tests as in the paper.
Go/no-go A1: median 3-shingle containment of synopsis in its own narrative
at most 0.8 (above that the synopsis is an extract, not a view, and the leg
is dropped); median synopsis length at least 15 tokens.

A2, reporter 1 vs reporter 2. Dual-report events only: one model, trained
on primary narratives as in the paper, evaluated twice on each dual-report
held-out record, once per narrative. This holds the event, label, stage and
model fixed and varies only whose account is read. The training domain
favours reporter 1 and the subset is multi-crew by construction; both are
stated, not corrected.
Go/no-go A2: at least 1500 dual-report records inside the task with both
narratives of usable length, and at least 400 of them in the held-out
partition.

## Leg B: NHTSA recall campaigns (new, second industry)

Three authored fields per campaign (defect description, consequence,
corrective action), component category as target, collapsed to its top
level. Campaign-level dedup; split grouped by campaign; configurations
TF-IDF + BiLSTM over task-trained word2vec; three seeds.
Go/no-go B: the flat file carries the three text fields as described; at
least 8 component classes with 300+ campaigns after dedup; median field
lengths at least 15 tokens each. If B fails, the Eclipse defect dataset may
be assessed as a substitute under the same criteria, once; if that also
fails, leg B is dropped, not replaced.

## Outcome commitments, unchanged

All cells report as measured. Clinical note-type findings enter the
Discussion as independently converging literature regardless of outcomes.
No further datasets beyond these two legs under any outcome.
