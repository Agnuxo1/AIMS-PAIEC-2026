"""Beta-Binomial shrinkage predictor for the PAIEC.

Submission for the Predictive AI Evaluation Competition at NeurIPS 2026.

Method
------
Predict the probability that a subject answers an unobserved item correctly as
the posterior mean of a Beta-Binomial model for the subject-benchmark pair:

    p = (a + k) / (a + b + n),   a = P0 * S,  b = (1 - P0) * S

where k is the number of successes among the n revealed labels for that pair,
P0 = 0.5 is a symmetric prior and S = 3 is the prior strength: the argmin of the
development split of 50 held-out benchmarks, inside a region that is flat over
S in [3, 5] (0.21708, 0.21740, 0.21809). It was never tuned on the benchmarks
used for the reported results.

Why this and not something richer
---------------------------------
The evaluation is a benchmark-level cold start: the items come from benchmarks
absent from training, so there are no item or benchmark parameters to transfer.
We measured, under leave-one-benchmark-out validation over 101 held-out public
benchmarks, that neither a subject-identity prior (46/101 benchmarks, sign-flip
p = 0.43) nor a non-parametric prior fitted to the historical distribution of
pair accuracies (47/101, p = 0.55) improves reliably over this symmetric prior.
Essentially all of
the gain over the official empirical-mean reference comes from calibrating the
shrinkage, not from a more informative prior. The technical report documents
those negative results in full.

No state is kept between calls, nothing is fitted at run time, no network or
file access is required, and the function is pure given its arguments.
"""

PRIOR_MEAN = 0.5
PRIOR_STRENGTH = 3.0
EPSILON = 1e-6


def predict(input: list[dict], labeled: list | None = None) -> float:
    """Return P(response = 1) for the target subject-item pair.

    `input` is [subject, item]; `labeled` holds entries [[subject, item], y]
    revealed within the current budget. Only labels from the same subject and
    the same anonymous benchmark_id are used, matching the quantity the metric
    averages over.
    """
    subject, item = input
    benchmark = item.get("benchmark_id") if isinstance(item, dict) else None

    # If the platform ever omits benchmark_id we fall back to the prior rather
    # than pooling evidence across benchmarks, which would mix unrelated pairs.
    if not (isinstance(benchmark, str) and benchmark):
        return float(PRIOR_MEAN)

    successes = 0
    count = 0
    for entry in labeled or ():
        try:
            (observed_subject, observed_item), label = entry
        except (TypeError, ValueError):
            continue
        if observed_subject != subject:
            continue
        if not isinstance(observed_item, dict):
            continue
        if observed_item.get("benchmark_id") != benchmark:
            continue
        if label in (0, 1, 0.0, 1.0):
            successes += int(label)
            count += 1

    a = PRIOR_MEAN * PRIOR_STRENGTH
    b = (1.0 - PRIOR_MEAN) * PRIOR_STRENGTH
    p = (a + successes) / (a + b + count)
    return float(min(max(p, EPSILON), 1.0 - EPSILON))
