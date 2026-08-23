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
