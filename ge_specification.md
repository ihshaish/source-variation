# GE analyses specified on 15 November 2026, before they were run

Written before the analysis scripts were sent to GE and before any result of
these analyses existed.

1. **Primary uncertainty.** Paired bootstrap with 10,000 draws over the 1,159 held-out events. Each draw is shared by both conditions and all three runs. The estimate is the mean of the within-run differences; the interval is conditional on the trained models and the reference partition. Two-sided approximate randomisation with 10,000 permutations: one swap per event, shared across runs; statistic = |mean difference|; plus-one correction. Holm adjustment within each family: headline {H1, H2}, model by field {M1–M3}, masking {K1–K5}, split {S1–S3}, lineage {R1}. A unit-cluster bootstrap is reported as a dependence check. The range over runs is reported separately.
2. **Headline contrasts.** H1 = technician minus customer (Avi2Vec-BiLSTM). H2 = repair-action GloVe-200 BiLSTM minus GloVe-200 mean pooling. Both use the published runs.
3. **Primary mask.** The engineer-completed v2 list, containing the original list, applied identically to all fields, classes and partitions. Two variants: full (names, no-fault terms, replacement verbs) and names-only. Results are reported whatever they show.
4. **Robustness split.** Chronological split as primary, because the deployment question is later returns from the same fleet, where unit overlap is expected. Unit-grouped is secondary and is reported with the unit-overlap audit. No extra seeds on the reference split.
5. **Not pursued.** The residual S(technician, m0) − S(customer, m1): no validation-selected customer model exists, so it would be post hoc.

Check on archived published arrays (15 Nov 2026, computed outside GE from the exported prediction arrays):
H1 +0.4562 [+0.4221, +0.4888], p = 0.0001; H2 +0.0924 [+0.0707, +0.1149], p = 0.0001 (raw; Holm over the two = 0.0002).
