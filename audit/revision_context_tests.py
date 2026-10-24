"""Lightweight synthetic API/cohort checks. Never reads context scores or logs."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate_context as ec


def main():
    root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location('context_under_test', root / 'context_model.py')
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    columns = ['normalized_name', 'provider', 'release_date', 'access_date', 'harness',
               'harness_version', 'reasoning_effort', 'subject_features_extra']
    # Seven hidden subjects, with two identical visible signatures. Groups of four
    # therefore also exercise the final cohort of three; size eight uses seven.
    pairs = pd.DataFrame([
        {column: ('subject-' + str(i if i != 1 else 0) if column == 'normalized_name' else '')
         for column in columns} for i in range(7)
    ])
    scenarios = pd.DataFrame([
        {'benchmark_id': 'synthetic', 'seed': seed, 'pair_index': i}
        for i in range(7) for seed in [17, 29]
    ])
    rng = np.random.default_rng(72)
    labels = rng.integers(0, 2, size=(len(scenarios), 31))
    cumulative = np.column_stack([np.zeros(len(scenarios), dtype=int), np.cumsum(labels, axis=1)])
    successes = cumulative[:, ec.N]
    max_error = 0.0
    checks = 0
    for size in (1, 2, 4, 8):
        evidence_summary = ec.cohort_evidence(pairs, scenarios, successes, size)
        for mix in (0, .25, .5, 1):
            for strength in (1, 3, 8, 16):
                expected = ec.prediction(evidence_summary, mix, strength)
                model.PEER_MIX = mix
                model.PRIOR_STRENGTH = strength
                for seed, seed_rows in scenarios.groupby('seed'):
                    ids = np.random.default_rng(ec.stable('peer-cohort', 'synthetic', int(seed))).permutation(seed_rows.index.to_numpy())
                    for start in range(0, len(ids), size):
                        group = ids[start:start + size]
                        for target in group:
                            subject = pairs.loc[scenarios.at[target, 'pair_index']].to_dict()
                            item = {'benchmark_id': 'synthetic'}
                            for j, budget in enumerate(ec.N):
                                evidence = [
                                    [[pairs.loc[scenarios.at[peer, 'pair_index']].to_dict(), {'benchmark_id': 'synthetic'}], int(y)]
                                    for peer in group for y in labels[peer, :budget]
                                ]
                                actual = model.predict([subject, item], evidence)
                                max_error = max(max_error, abs(actual - expected[target, j]))
                                checks += 1
                                assert abs(actual - expected[target, j]) < 1e-14
    print('SYNTHETIC_CONTEXT_API_PARITY cases', checks, 'maxerr', max_error,
          'sizes1/2/4/8 tail3/7 and visible-collision included')

    model.PRIOR_STRENGTH = 3
    model.PEER_MIX = .5
    for budget in range(32):
        for correct in range(budget + 1):
            subject = {'normalized_name': 's'}
            item = {'benchmark_id': 'b'}
            evidence = [[[subject, item], int(i < correct)] for i in range(budget)]
            assert model.predict([subject, item], evidence) == (1.5 + correct) / (3 + budget)
    print('FALLBACK_ALL_n0to31_k PASS', sum(budget + 1 for budget in range(32)))

    actual_pairs = pd.read_parquet(root / 'pairs.parquet')
    folds = actual_pairs.groupby('benchmark_id').outer_fold.first().to_dict()
    for fold in range(5):
        roles = json.loads((root / f'fold-{fold}-roles.json').read_text())
        fit, validation, test = (set(roles[key]) for key in ['fit', 'validation', 'test'])
        assert not (fit & validation or fit & test or validation & test)
        assert all(folds[benchmark] == fold for benchmark in test)
        assert all(folds[benchmark] != fold for benchmark in validation)
    print('ROLE_MANIFESTS5 PASS disjoint; validation excludes exterior')

    columns = ['normalized_name', 'provider', 'release_date', 'access_date', 'harness',
               'harness_version', 'reasoning_effort', 'subject_features_extra']
    signatures = actual_pairs[columns].fillna('').astype(str).apply(
        lambda row: json.dumps(row.to_dict(), sort_keys=True), axis=1)
    counts = actual_pairs.assign(signature=signatures).groupby(['benchmark_id', 'signature']).size()
    collisions = counts[counts > 1]
    print('WITHIN_BENCH_COLLISIONS', len(collisions), 'pairs', int(collisions.sum()),
          'counts', collisions.tolist())
    for filename in ['EXPERIMENT_D.md', 'evaluate_context.py', 'context_model.py']:
        print('SOURCE_SHA256', filename, hashlib.sha256((root / filename).read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
