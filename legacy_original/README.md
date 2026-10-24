# Shrinkage under benchmark holdout

Submission for the **Predictive AI Evaluation Competition at NeurIPS 2026**
(Stanford AIMS) — <https://aims.stanford.edu/competition>

**Francisco Angulo de Lafuente**, independent researcher, Madrid, Spain.

---

## What this is

Predict the probability that an AI system answers a previously unevaluated item
correctly, scored by Brier ALC over label budgets 0, 1, 3, 7, 15 and 31, computed
per subject–benchmark pair and averaged with every pair weighted equally. The
evaluation is a **benchmark-level holdout**: held-out items come from benchmarks
absent from training.

The submitted predictor is the posterior mean of a Beta–Binomial model:

```
p = (a + k) / (a + b + n),   a = P0·S,  b = (1−P0)·S,   P0 = 0.5,  S = 3
```

where `k` is the number of successes among the `n` labels revealed for that
subject and that `benchmark_id`. Standard library only, keeps no state, fits
nothing at run time, needs no payload. `S = 3` is the development argmin among
the Beta–Binomial variants in `src/barrido_prior.py`, which writes `hiper.json`;
the optimum is flat over S ∈ [3,5] (0.21708 / 0.21740 / 0.21809).

## Headline result

Leave-one-benchmark-out over **101 held-out public benchmarks** (496 pairs),
default acquisition order (`src/ejecuta.py final`):

| Predictor | Brier ALC ↓ | B₀ | B₁ | B₃ | B₇ | B₁₅ | B₃₁ |
|---|---|---|---|---|---|---|---|
| Empirical-Bayes, subject-conditioned | 0.2099 | 0.2471 | 0.2311 | 0.2119 | 0.1991 | 0.1905 | 0.1865 |
| **Beta–Binomial, S=3 (submitted)** | **0.2105** | 0.2500 | 0.2319 | 0.2128 | 0.1992 | 0.1905 | 0.1863 |
| Empirical mean (**official reference**) | 0.2454 | 0.2500 | **0.3650** | 0.2434 | 0.2076 | 0.1925 | 0.1868 |
| Constant 0.5 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 |
| *Oracle, one value per pair* | *0.1789* | | | | | | |

A relative reduction of **14.2 %** (absolute 0.0349) against the official
reference. On a run with randomised acquisition order, resampling benchmarks with
their pairs kept together gives **−0.0310, 95 % CI [−0.0373, −0.0249]**. Improves
**86 of 101** benchmarks; median per-benchmark difference −0.0405; stable across
five acquisition draws (s.d. 0.0005); between 13.1 % and 14.5 % across every
sampling choice varied; **−0.0323** when benchmarks are reweighted by their
eligible pair counts; **−0.0317** after dropping the five benchmarks where the
gain is largest.

Two caveats up front:

- The official reference is **deliberately simple** — it returns the raw mean of
  revealed labels, so at one label it predicts 0 or 1 and a wrong guess costs the
  maximum Brier loss. Its B₁ of 0.3650 is worse than ignoring the labels. Beating
  it is not evidence of being competitive.
- Its own advantage over predicting 0.5 is **not reliable** across benchmarks
  (+0.0094, 95 % CI [−0.0064, +0.0252] on the randomised-order run).

An oracle restricted to one value per pair scores 0.1789, so we capture **52.5 %**
of the headroom available to any pair-constant predictor. Of that floor, **96.7 %
is variation between items inside a pair** and only 3.3 % is repetition noise.

## What did not work

Reported in full in the paper, because it bounds where the remaining signal
is *not*.

| Attempt | Result |
|---|---|
| Response-weighted global prior (0.350) | **Worse**: ALC 0.2137, B₀ 0.2650. The metric averages per pair (0.538 among the eligible rows), not per response. |
| Subject-identity prior | Helps B₀ (0.2471 vs 0.2500) but ALC difference −0.0005, CI [−0.0024, +0.0012], p = 0.64 *(randomised-order run; 46/101 benchmarks, p = 0.43 with the default order)*. |
| Non-parametric prior over the bimodal pair-accuracy distribution | +0.0001, p = 0.81 *(randomised-order run; 47/101, p = 0.55 with the default order)*, across filters, recentring and mixing weights. |
| Per-budget shrinkage tuning | Selects S=3 at every budget with labels; identical to a single S=3. |
| Logistic model on item-text + subject features (23 columns) | B₀ 0.2495 vs 0.2500, wins in 18/50 development benchmarks, p = 0.065, worth −0.00005 ALC. |
| **Per-item deviation from the pair mean** (the largest target: 96.7 % of the floor) | **Strictly worse at every weight**: development ALC 0.21708 → 0.21776 → 0.21896 → 0.22067 at λ = 0, 0.5, 1.0, 1.5. |
| Same, with hashed sparse text (2¹⁸ word + 2¹⁷ char features), **starting from the true pair mean** | Oracle 0.1905 → 0.1907 (λ=0.5) → 0.1912 (λ=1.0); improves only 14/50 benchmarks, p = 0.003 *against* the addition; mean correlation between predicted and realised deviation −0.016. |

The organisers' companion embedding release `measurement-db-embed` covers only 3
of the 165 binary benchmarks, too few to measure transfer, so that route is
closed for now. A learned representation from a language model, and
inference-time API calls, remain untested; the 0.1730 of between-item variance
remains available to whoever can reach it.

## Audit of the public release

Definitions fixed before looking at the data. Nothing uses platform feedback; the
rule against identifying anonymous subject or benchmark identifiers was
respected. All figures cover the 165 binary public benchmarks.

- **Confirmed — items shared across benchmarks.** 1,980 of 2,898,062 distinct
  content hashes (0.068 %) appear in more than one benchmark, across 11
  benchmarks; largest overlap `adaptivestep`↔`prox` (1,819 items). Irrelevant to
  our submitted predictor, which never reads item content, but not in general:
  any method using item text could count the same item twice. Whether any such
  item also sits in a *private test* benchmark is the case that would matter and
  cannot be checked from outside — raised as a question for the organisers.
- **Not found — leakage into predictor-visible fields.** 2 of 60 benchmarks
  screened had keyword matches in `item_features`; neither coincided with the
  grading criterion. No finding in a keyword screen over 60 of 165 benchmarks,
  which is not evidence of absence.
- **Confirmed — a fifth of repeated observations contain both outcomes.**
  635,977 of 3,154,268 repeated subject–item pairs (20.2 %; median 8.3 % per
  benchmark) contain both a success and a failure, some repeated up to 10,000
  times. This does **not** by itself explain the 0.1789 floor: repetition
  variance accounts for only 0.0059 of it.

## Reproducing

Python 3.10+, `pandas`, `pyarrow`, `numpy`, `scikit-learn`, `huggingface_hub`,
about 14 GiB of disk, a few hours on one desktop CPU, no GPU. Data: the public
release at verified data commit `a23e041be8651be4d97decba385779300abeabee`,
builder revision `287495967fc276275c3b466dfede2cc3534e830c`.

```bash
python src/descarga_datos.py responses,subjects,benchmarks   # ~2.5 GiB
python src/descarga_datos.py items                           # ~11.2 GiB
python src/test_metrica.py          # 10 checks on the metric simulator
python src/construye.py             # evaluation set: 769 pairs, 151 benchmarks
python src/estadisticas_datos.py    # descriptive statistics cited in the paper
python src/barrido_prior.py         # prior variants, development only -> writes hiper.json
python src/barrido_presupuesto.py   # per-budget shrinkage sweep, development only
python src/ejecuta.py tune          # shrinkage curve for the hierarchical variants
python src/ejecuta.py final         # headline table, 101 held-out benchmarks
python src/robustez.py              # seeds, both averaging units, bootstrap, headroom
python src/sensibilidad.py          # sampling sensitivity, oracle decomposition, reweighting
python src/auditoria.py             # the three audit findings
python src/texto_b0.py              # transferable text signal at budget 0
python src/por_item.py              # per-item deviation, hand-built features
python src/tfidf_item.py            # per-item deviation, hashed sparse text
python hacer_zip.py                 # builds submission.zip
```

`src/barrido_prior.py` is what sets the submitted shrinkage strength: it writes
`hiper.json`, which `robustez.py`, `sensibilidad.py` and `por_item.py` read.
`src/ejecuta.py tune` sweeps the hierarchical variants and does **not** write it,
so the order above matters.

The scripts use absolute paths under `E:\AIMS-sprint\` (`DATA`, `W`) and
`barrido_prior.py` writes `hiper.json` into the working directory; run them from
the directory holding `pares.pkl` and `agregados.parquet`, or edit those
constants for another layout.

All partitions derive from `blake2b`, not Python's `hash`, which is randomised
per process. An earlier version used the latter and an independent re-derivation
caught it; every number above was regenerated afterwards.

Access to `aims-foundations/measurement-db` on Hugging Face requires accepting
its conditions.

## Layout

```
submission/model.py   the submitted predictor (this is the whole submission)
src/                  harness, predictors, LOBO runner, robustness, audit, tests
resultados/           every JSON/CSV the paper cites
informe/informe.tex   the technical report
hacer_zip.py          builds submission.zip from submission/
```

## Disclosure

Training and development used only the public `measurement-db` release listed in
the benchmark inventory. No other data source was used. The work was carried out
with substantial assistance from large language models, for code, for prose and
as adversarial critics of the analysis; three rounds of independent re-derivation
were run before submission and each found errors that were corrected.

## Licence

MIT.
