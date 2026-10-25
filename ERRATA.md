# Errata and known deviations

This file lists what is wrong, unverified or only partly reproducible in the
published material, and what was done about it. It is written to be read
against the README, the report and the submission archives.

**Two rules applied.** Nothing already deposited with the organisers has been
edited: the report PDF (SHA-256 `5510cd7…`) and both ZIP archives are unchanged,
because changing them would mean sending a new version outside, which is a
decision for the author. Corrections therefore live here and in the README.
Where a statement below comes from the independent review and I did not
reproduce it myself, it is marked as such.

Last update: 2026-10-25.

Correction in report v3: item 4's unsupported original-protocol bootstrap interval is withdrawn.
`report/informe_v2.pdf` remains the historical deposited file; the corrected report is
`report/informe_v3.pdf`. Item 5 is a docstring erratum in the unchanged ZIP, not a functional change.

---

## Corrected in this update

1. **Download command.** The README ran `audit/download_pinned.py` without
   `--data`, which the script requires, so a clean clone failed. Fixed in the
   README.

2. **Official checker.** The statement that both archives pass the published
   `check_submission_zip.py` had no log behind it. Both were run on 2026-10-25
   and return status 0 with `OK`. Logs are in `docs/evidence/`.

3. **Figure 4 hard-coded numbers.** `build/make_figures.py` typed `0.2368`,
   `0.2060`, `133/151` and `-0.0026` into the figure, which contradicted the
   claim that no figure number is typed by hand. The figure now reads its
   numbers from `results/`. The image is unchanged in content.

---

## Not corrected: deposited material, needs the author's decision

4. **Unsupported bootstrap in the report.** `report/informe_v2.tex` (around line
   151) quotes an "independent 10,000-benchmark bootstrap" with interval
   `[-0.04107, -0.02854]`. No script or result file in the repository produces
   it. The phrase "10,000-benchmark" is also loose: it means 10,000 resamples of
   the 101 benchmarks. Treat the interval as unsupported until a generator is
   committed.

5. **Mislabelled test in the submitted docstring.** The docstring of
   `submission_original.zip/model.py` refers to a "sign-flip" p = 0.43. The
   test actually used is an exact binomial sign test. The report says so
   (around line 153). The archive itself is left as deposited.

---

## Reproducibility gaps

6. **Code changed after some scores were computed.** The list is in
   `results/final_code_changes.json`. Among the changes: an IC99 interval for
   five candidates in place of the earlier interval, a cache-semantics guard,
   and a changed way of locating the original model. The README did not say so
   before this file.

7. **Hashes not recorded in the locks.** `audit/build_independent.py`,
   `audit/evaluate_independent.py` and `audit/evaluate_context.py` do not appear
   in any preregistration or lock file. Only `audit/context_model.py` matches its
   recorded lock (`64d53e…`). The results were produced by earlier versions of
   these scripts, so a clean run of the current scripts is not guaranteed to
   reproduce every JSON byte for byte.

8. **Result files without a generator in the repository.** These have no script
   here that writes them: `results/final_context_config.json`,
   `results/provenance_validation.json`, `results/sampling_endpoint_comparison.json`,
   `results/aggregate_cache_comparison.parquet`, `results/all_pair_aggregates.parquet`,
   and the supporting and historical manifests.

9. **The original-protocol replication is not portable.**
   `audit/replicate_original_design.py` reads `E:\AIMS-sprint\work\*.parquet`
   and a catalogue file that are not in the repository. The replication
   `0.2453845 → 0.2104970` cannot be rerun from a clean clone as shipped.

---

## Choices that need stating plainly

10. **The contextual candidate's ρ = 0.5 was not fixed in advance.** It was
    already the research default in `audit/context_model.py` when the
    preregistration was locked, and `EXPERIMENT_D.md` gives no rule for picking a
    single final ρ. The independent review reports that an unweighted mean
    across folds would select ρ = 1.0 (0.20519 against 0.20546). I did not
    reproduce that figure. The IC99 for five candidates was not preregistered:
    the preregistration specified an IC98.75 for four families.

11. **The submitted S = 3 was chosen on the earlier development split.** It is
    the argmin on that split, documented in `results/hiper.json`. The independent
    review reports that under the new nested validation S = 3 wins in only one
    of five folds. I did not reproduce this. The headline figures of the
    submitted predictor are the earlier protocol's; the new protocol's figures
    are separate and labelled as such.

12. **"No leakage found" has a narrow scope.** The keyword screen covered 60 of
    the 165 binary benchmarks, and its outcome in `results/auditoria.json` is
    "INDICIO SIN CONFIRMAR", not a clean result. It is evidence of no keyword
    hit in those 60, not of absence.

13. **Mixed protocols in the README tables.** Some rows in the negative-results
    table come from the earlier randomised-order protocol and some from the new
    protocol. The review flags rows that do not say which. Read each number with
    the protocol named in its row.

---

## Not affected

- The headline figures of the new protocol (0.236814 → 0.205984, 133/151,
  interval `[-0.038426, -0.027214]`) were re-derived from the pair-level parquet
  files by the independent review and match `results/summary.csv` and
  `results/uncertainty.json`.
- No reported figure is presented as an official score, a ranking or an
  acceptance. The contextual extension is marked experimental and conditional on
  an availability that is not verified.
- The rule that the sprint must not use Python's `hash()` for partitions holds
  in the current scripts: seeds come from SHA-256 or blake2b.
