# E1 — Same-item peer labels (pre-registration, 2026-10-28 ~08:30 Madrid)

## Hypothesis
H1: When `labeled` contains a label for the *same item* from another subject of the
same benchmark (possible because the 50/50 acquisition/evaluation split is drawn
per subject–benchmark pair), shifting the pair-level prediction by that peer's
residual lowers Brier ALC versus the control.

H0: no improvement (difference >= 0), or an improvement whose 95% component
bootstrap interval includes 0.

## Why it might work
Within-pair item variation is the largest unexplained component (the per-pair
oracle gap). Item difficulty is shared across subjects (IRT intuition), and the
official ingestion (`tools/streaming_ingestion.py`, baseline commit cc88a95) passes
one global `labeled` list containing every acquired label of every pair in the
sample. Codabench submission 1013718 showed 2 of 7 benchmarks with two subjects.

## Design (platform-like cohort simulator)
- Data: per-(subject,item) success/trial aggregates of the pinned public release
  (same cache as the independent protocol), eligible pairs = >= 80 distinct items.
- Benchmarks with >= 2 eligible subjects. For each of 8 fixed seeds
  (17,29,43,71,101,137,181,233) and each benchmark: up to 3 disjoint cohorts of
  m subjects (primary m = 2, as observed on Codabench; secondary m = 4).
- Item sample: items attempted by every cohort subject; T = min(|common|, U),
  U ~ Uniform{88..152} (formative pairs had 44–76 evaluation items => 88–152 total).
  Cohorts with < 80 common items are skipped.
- Each pair: independent random 50/50 split; random acquisition = first 31 of the
  acquisition pool; revealed label ~ Bernoulli(item success rate) (one recorded response).
- Budgets 0,1,3,7,15,31, nested; labeled at budget n = first n acquired labels of
  every pair in the cohort. Brier over every recorded evaluation response;
  ALC weights 0.1,0.2,0.2,0.2,0.2,0.1; pairs averaged per cohort, cohorts per benchmark,
  benchmarks equally.

## Predictors
- CONTROL A: submitted predictor, Beta(1.5,1.5) posterior mean of own pair labels.
- CONTROL B: context_experimental (peer pair evidence in the prior, mix 0.5, S 3).
- E1(base, lam, c): base in {A, B};
  p_i = clip(base + lam * sum_j (y_ji - phat_j^(-i)) / (c + count_i)),
  phat_j^(-i) = leave-one-out Beta(1.5,1.5) mean of peer j; lam in {0.25,0.5,0.75,1},
  c in {0,1,3}. Identical to base when no same-item peer label exists.

## Selection and test
Leave-one-outer-fold-out over the 5 existing component folds (benchmarks connected by
shared content hashes never cross folds): choose (lam, c) on 4 folds, score the 5th.
Primary metric: held-out ALC difference E1(A) - A, mean over benchmarks.
Accept iff the 95% bootstrap interval (10,000 resamples of components) is < 0 AND the
mean difference is < 0 in at least 6 of 8 seeds. Negative results are recorded.
A Codabench upload is made only if accepted; Codabench (9 pairs, SE ~0.022) cannot
confirm or reject an effect of this size and is used only as an execution check.
