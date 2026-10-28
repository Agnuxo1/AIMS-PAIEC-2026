# Experiment log

Every entry follows hypothesis -> design -> control -> run (fixed seeds) -> official metric
(Brier ALC) -> decision. Negative results stay. Machine-readable copy: `experiments.csv`.

## E0 family (24 Oct 2026) — see `results/` and `report/`
Independent protocol over 151 public benchmarks, 13,200 pairs, 8 seeds, grouped by content
components. The submitted Beta-Binomial rule beats the official empirical-mean reference by
-0.030830 (CI95 [-0.038426, -0.027214]). Tuned prior and calibration table: inconclusive.
Subject ridge prior: worse. Peer-benchmark context (CTX): -0.002562 in 4-subject cohorts.

## Codabench 1013718 (27 Oct 2026)
`submission_original.zip`. Formative Brier ALC **0.208365**, ECE 0.136929. The formative sample
holds only 9 subject-benchmark pairs (7 benchmarks; 2 benchmarks had 2 subjects), so the
standard error across pairs is ~0.022: an execution check, not a model comparison.
Per-pair feedback: `docs/evidence/codabench_1013718_formative.csv`.

## E1 — same-item peer residuals (28 Oct 2026)
- Hypothesis and acceptance rule pre-registered in `E1/PREREG_E1.md` before running.
- Evidence for the mechanism: the official ingestion (`paiec_baseline` cc88a95,
  `tools/streaming_ingestion.py`) passes one global `labeled` list to every evaluation call, and
  the acquisition/evaluation split is drawn per subject-benchmark pair, so a peer may have revealed
  the very item being predicted.
- Simulator `E1/run_e1.py` (cohorts of 2 primary, 4 secondary; items common to the cohort;
  88-152 items per pair; 50/50 split per pair; random acquisition; 8 seeds). Run on servidor-ia
  CPU (Python 3.13.5, numpy 2.5.4, pandas 3.1.0, pyarrow 26.0.0; `E1/requirements-e1.lock`).
- Selection: leave-one-outer-fold-out over the 5 component folds. The same configuration
  (base CTX, lambda 0.75, c 1) was chosen in all 5 folds for both cohort sizes.

| Comparison (held-out folds) | cohorts of 2 | cohorts of 4 |
|---|---|---|
| E1 vs submitted E0 | -0.004437 [-0.005139, -0.003741], 8/8 seeds, 118/134 benchmarks | -0.009450 [-0.010788, -0.008131], 8/8, 97/105 |
| E1 vs CTX | -0.001930 [-0.002305, -0.001546], 8/8, 116/134 | -0.005583 [-0.006602, -0.004598], 8/8, 92/105 |
| E1 without CTX prior vs CTX | +0.000459 [-0.000390, +0.001357], 0/8 — rejected | -0.001998 [-0.003542, -0.000355] |

Decision: **accepted**. Exact API parity of `submissions/item_peer/model.py` with the simulator
(216 pair curves, max error 0) and exact equality with the submitted predictor when no peer labels
exist (2,220 checks): `E1/parity_e1.py`.

Process note: the first analysis call failed on a column-name format (`0` vs `0.0`); the
simulation outputs were reused unchanged and only `analyse()` was re-run (`run_e1.py N analyse`).

Caveat: these simulations place every pair in a multi-subject cohort. On Codabench only part of
each sample shares a benchmark (4 of 9 pairs in 1013718), so the expected platform gain is
diluted roughly in proportion; with no peers the output is identical to E0.

## Codabench 1015136 (28 Oct 2026)
`submission_item_peer.zip` (E1). Formative Brier ALC **0.185082**, ECE 0.123419, on a new sample of
8 pairs over 7 benchmarks (one benchmark with 2 subjects). Not a paired comparison with 1013718:
the samples differ and each has SE ~0.022. Selected as the final entry on the strength of E1.
Per-pair feedback: `docs/evidence/codabench_1015136_formative.csv`.
