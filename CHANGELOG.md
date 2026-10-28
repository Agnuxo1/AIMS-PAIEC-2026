# v2.0.0, 28 October 2026

- Codabench reachable: team registered; submission 1013718 (original rule) finished with formative Brier ALC 0.208365.
- Verified from the official ingestion code that one global `labeled` list reaches every prediction.
- Experiment E1, pre-registered: same-item peer residuals on top of the benchmark-context prior. Accepted on held-out folds (cohorts of 2: -0.004437 vs the submitted rule, 8/8 seeds). Negative variant (residuals without the context prior) kept.
- New archive `submissions/submission_item_peer.zip`, exact parity with the simulator and exact fallback to the original rule; submitted as 1015136 and selected for final evaluation.
- Technical report v4 (7 pages, NeurIPS template) with the platform evidence, E1 and the official feedback.
- New figure and two GIFs (`fig5_e1_item_peer.png`, `gif3_item_peer.gif`, `gif4_codabench_alc.gif`); experiment log `experiments/LOG.md` and `experiments/experiments.csv`.
- CI also checks the E1 archive's fallback.

# v2, 24 October2026

- Preserve all original source/results and original ZIP.
- Reconstruct raw-data eligibility and exactly replicate the original headline without its evaluator/pickle.
- Replace capped sampling with full eligible pairs and full evaluation responses.
- Add grouped nested selection, per-component uncertainty and complete candidate logs.
- Test three isolated-evidence families and two contextual variants; keep failures.
- Add a zero-dependency contextual candidate with exact original fallback.
- Freeze890public input hashes, add pinned downloader and portable data location.
- Replace overbroad negative/oracle claims with bounded conclusions; five-page official-style report.
- Add direct synthetic data/API controls, independent review, MIT file and third-party terms.
