# MaintNet external test: registered note

Written before downloading any data; the git history carries the timestamp.
The question: does the source and target alignment observed on the GE
records recur on a public maintenance corpus? MaintNet's aviation logbooks
carry a problem narrative and an action narrative for the same maintenance
event, which is the structure the question needs.

## Design

Six cells: three inputs (problem text, action text, both concatenated) by
two target families (one problem-side label and one action-side label,
chosen from the annotated release by the support criteria below). The
headline contrast is problem text against action text on the action-side
target, which is the cross-stage cell that parallels the GE customer field
predicting the repair outcome. Diagonal cells, where a field predicts a
label annotated from that same field, are reported but discounted. The
control is target-span masking at evaluation, and the result is whatever
advantage survives the mask.

Three configurations: TF-IDF with logistic regression; BiLSTM over
GloVe-200; BiLSTM over word2vec trained on the task partition. No other
models, regardless of the results.

Split: stratified 80/20 held-out with near-duplicate confinement (MinHash
over 5-token shingles, Jaccard 0.8, as in the paper). Statistics:
record-level bootstrap intervals and paired approximate-randomisation
tests, Holm within families, three seeds. The same machinery as the ASRS
pipeline.

## Go or no-go criteria

- G1: the provenance of the annotated release can be verified and the
  licence permits use.
- G2: one problem-side and one action-side target, each with at least 4
  classes of at least 150 records after minimum-support filtering and
  near-duplicate collapse.
- G3: the action-side target's span can be identified, and so masked, in
  at least 80% of records.
- G4: at least 3000 effective events after near-duplicate collapse.
- Any failure means no-go. The paper is then submitted as it stands and no
  substitute dataset is sought.

## Outcome commitments

Whatever the six cells show is reported. Recurrence, crossover and a null
are all informative, and none of them triggers further experiments.

## Datasets considered and excluded

FAA SDR: a single narrative, so it cannot improve the identification of any
open claim. NTSB: the factual and cause narratives are finalised in one
report by one body, so the stage separation is editorial, and the domain
drifts to accident investigation. Open Repair Alliance: domain drift. MIMIC:
domain drift and access agreements. Further models, splits or tests on the
GE data: that axis is saturated. A blinded re-audit belongs to the follow-up
study the paper already specifies.

## Phase 0 outcome (recorded 2026-08-23): no-go

The annotated release (Zenodo 20779601, CC BY 4.0, 6169 records) was
inspected against the criteria above.

- The released ACTION_TYPE column is empty and PROBLEM_TYPE is 97% one
  value, so both advertised taxonomies are unusable as shipped.
- A verb-class taxonomy derived from the TAGGEDACTION spans passes the
  action-side support bar, barely: replace-install 4221, inspect-check 434,
  repair-secure 418, adjust 301 (a 69% majority class).
- The problem side fails G2. The fleet's logbook is topically homogeneous:
  LOCATION collapses to cylinder (3999) and powerplant (1028) with nothing
  else above 9 records, and PROBLEM_PART collapses the same way, the
  frequent parts being gaskets, covers and baffles. No problem-side taxonomy
  with four supported classes exists, so the input-by-target matrix cannot
  be built.
- G4 is doubtful too: 3531 unique problem texts, roughly 2800 groups under
  an approximate near-duplicate collapse, and the true figure under the
  full procedure would be lower.

Any failure is a no-go under the criteria, so the fallback applies: the
paper is submitted as the two-corpus study it is, and no substitute dataset
is sought. The one comparison that survives, predicting the action verb
class from problem against action text, was considered and declined; with a
69% majority class and a self-annotated target it would not strengthen the
paper.

# Addendum (registered before tabulation): matched records inside the corpora we already hold

The MaintNet no-go settled the candidates reviewed so far; the question
itself stayed open. A wider survey, on the criteria of several records per
event, stated target provenance, public availability, scale after
de-duplication, records that are not derived from one another, and a
distinct contribution to the study, leaves two experimental parts, and no
more than two regardless of the results.

## Part A: ASRS matched records (data already in the paper)

A1, narrative against synopsis. The same events, labels and split as the
existing Aircraft task. The reporter's narrative is written before analyst
coding and the synopsis by the analyst who codes. Configurations: TF-IDF
with logistic regression; BiLSTM over GloVe-200; BiLSTM over word2vec
trained on the training partition of the respective record. Three seeds and
paired held-out tests as in the paper.
Go or no-go for A1: the median 3-shingle containment of the synopsis in its
own narrative is at most 0.8 (above that the synopsis would count as an
extract of the narrative and the part is dropped), and the median synopsis
length is at least 15 tokens.

A2, reporter 1 against reporter 2. Dual-report events only: one model,
trained on primary narratives as in the paper, evaluated twice on each
dual-report held-out record, once per narrative. This holds the event, the
label, the stage and the model fixed and varies only whose account is read.
The training domain favours reporter 1 and the subset is multi-crew by
construction; both facts are stated in the paper and neither is corrected
for.
Go or no-go for A2: at least 1500 dual-report records inside the task with
both narratives of usable length, and at least 400 of them in the held-out
partition.

## Part B: NHTSA recall campaigns (a second industry)

Three authored fields per campaign (defect description, consequence,
corrective action), with the component category, collapsed to its top
level, as the target. Campaign-level de-duplication; a split grouped by
campaign; TF-IDF and a BiLSTM over task-trained word2vec; three seeds.
Go or no-go for B: the flat file carries the three text fields as
described; at least 8 component classes with 300 or more campaigns after
de-duplication; median field lengths of at least 15 tokens each. If B
fails, the Eclipse defect dataset may be assessed once as a substitute under
the same criteria; if that also fails, part B is dropped and nothing
replaces it.

## Outcome commitments, unchanged

All cells are reported as measured. Findings on clinical note types enter
the Discussion as independently converging literature whatever the
outcomes. No further datasets beyond these two parts under any outcome.
