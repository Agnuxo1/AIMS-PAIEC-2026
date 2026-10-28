"""Benchmark-context shrinkage plus same-item peer residuals (experiment E1).

Uses ONLY the evidence supplied in the current call. The platform passes one
global `labeled` list; because the 50/50 acquisition/evaluation split is drawn
per subject-benchmark pair, another subject may have revealed a label for the
very item being predicted. That label says something about the item's
difficulty, measured as the peer's residual against its own leave-one-out
pair rate.

    base  = (k_own + S * m) / (n_own + S),   m = 0.5 + MIX * (peer_rate - 0.5)
    p     = base + LAM * sum_j (y_j - rate_j^(-item)) / (C + count)

With no peer labels on the benchmark this is exactly Beta(1.5, 1.5), the
predictor of submission 1013718. LAM and C were selected by
leave-one-fold-out validation over grouped public benchmarks (E1, see
experiments/E1). No cross-call state, network, files, or hidden IDs.
"""
import json

PRIOR_STRENGTH = 3.0
PEER_MIX = 0.5
ITEM_LAMBDA = 0.75
ITEM_C = 1.0
EPSILON = 1e-6


def _key(d):
    return json.dumps(d, sort_keys=True, separators=(",", ":"), default=str)


def predict(input: list[dict], labeled: list | None = None) -> float:
    subject, item = input
    benchmark = item.get("benchmark_id") if isinstance(item, dict) else None
    if not isinstance(benchmark, str) or not benchmark:
        return 0.5
    target_item = _key(item)

    own_k = own_n = 0
    peers = {}  # subject key -> [k, n, labels on the target item]
    for entry in labeled or ():
        try:
            (other, observed_item), y = entry
        except (TypeError, ValueError):
            continue
        if not isinstance(observed_item, dict) or observed_item.get("benchmark_id") != benchmark:
            continue
        if y not in (0, 1, 0.0, 1.0):
            continue
        y = int(y)
        if other == subject:
            own_k += y
            own_n += 1
            continue
        stats = peers.setdefault(_key(other), [0, 0, []])
        stats[0] += y
        stats[1] += 1
        if _key(observed_item) == target_item:
            stats[2].append(y)

    peer_k = sum(s[0] for s in peers.values())
    peer_n = sum(s[1] for s in peers.values())
    peer_mean = (1.5 + peer_k) / (3.0 + peer_n)
    prior = 0.5 + PEER_MIX * (peer_mean - 0.5)
    p = (own_k + PRIOR_STRENGTH * prior) / (own_n + PRIOR_STRENGTH)

    residual = 0.0
    count = 0
    for k, n, same_item in peers.values():
        for y in same_item:
            residual += y - (k - y + 1.5) / (n - 1 + 3.0)
            count += 1
    if count:
        p += ITEM_LAMBDA * residual / (ITEM_C + count)
    return float(min(max(p, EPSILON), 1.0 - EPSILON))
