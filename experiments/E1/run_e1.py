"""E1: same-item peer labels under a platform-like cohort simulator (see PREREG_E1.md)."""
from __future__ import annotations
import hashlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE / 'raw_cache'
SEEDS = (17, 29, 43, 71, 101, 137, 181, 233)
N = np.array([0, 1, 3, 7, 15, 31]); W = np.array([.1, .2, .2, .2, .2, .1])
LAMS = (0.25, 0.5, 0.75, 1.0); CS = (0, 1, 3)
EPS = 1e-6


def stable(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts, separators=(',', ':')).encode()).digest()[:8], 'little')


def pair_brier(p, succ, trials):
    return float(np.sum(trials * p * p - 2 * p * succ + succ) / trials.sum())


def simulate(bench, items_by_subject, subjects, m, seed):
    """Yield cohort records: per pair, eval arrays and acquired labels."""
    rng = np.random.default_rng(stable('E1', bench, m, seed))
    order = list(rng.permutation(subjects))
    cohorts = []
    while len(order) >= m and len(cohorts) < 3:
        group, order = order[:m], order[m:]
        common = set.intersection(*(set(items_by_subject[s].index) for s in group))
        if len(common) < 80:
            continue
        common = np.array(sorted(common))
        T = min(len(common), int(rng.integers(88, 153)))
        chosen = rng.choice(common, size=T, replace=False)
        pairs = []
        for s in group:
            tab = items_by_subject[s].loc[chosen]
            perm = rng.permutation(T); acq, ev = perm[:T // 2], perm[T // 2:]
            acquired = acq[:31]
            rate = (tab['successes'].to_numpy() / tab['trials'].to_numpy())
            y = (rng.random(len(acquired)) < rate[acquired]).astype(np.int64)
            pairs.append({'subject': s, 'acq_items': chosen[acquired], 'y': y,
                          'ev_items': chosen[ev], 'ev_succ': tab['successes'].to_numpy()[ev],
                          'ev_trials': tab['trials'].to_numpy()[ev].astype(float)})
        cohorts.append(pairs)
    return cohorts


def cohort_scores(pairs):
    """Return dict config -> list of per-pair ALC."""
    out = {}
    for j, P in enumerate(pairs):
        curves = {}
        for bi, n in enumerate(N):
            # own evidence
            nj = min(n, len(P['y'])); kj = int(P['y'][:nj].sum())
            base_a = (kj + 1.5) / (nj + 3.)
            pk = pn = 0
            for jj, Q in enumerate(pairs):
                if jj == j: continue
                nq = min(n, len(Q['y'])); pk += int(Q['y'][:nq].sum()); pn += nq
            peer = (1.5 + pk) / (3. + pn); mprior = .5 + .5 * (peer - .5)
            base_b = (kj + 3 * mprior) / (nj + 3)
            # same-item peer residuals for every eval item
            rsum = np.zeros(len(P['ev_items'])); rcnt = np.zeros(len(P['ev_items']))
            pos = {it: i for i, it in enumerate(P['ev_items'])}
            for jj, Q in enumerate(pairs):
                if jj == j: continue
                nq = min(n, len(Q['y'])); kq = int(Q['y'][:nq].sum())
                for it, yv in zip(Q['acq_items'][:nq], Q['y'][:nq]):
                    i = pos.get(it)
                    if i is None: continue
                    loo = (kq - yv + 1.5) / (nq - 1 + 3.)
                    rsum[i] += yv - loo; rcnt[i] += 1
            succ, tr = P['ev_succ'], P['ev_trials']
            cfgs = {('A', 0, 0): np.full(len(tr), base_a), ('B', 0, 0): np.full(len(tr), base_b)}
            for lam in LAMS:
                for c in CS:
                    shift = np.where(rcnt > 0, lam * rsum / np.maximum(c + rcnt, 1e-9), 0.)
                    cfgs[('A', lam, c)] = base_a + shift
                    cfgs[('B', lam, c)] = base_b + shift
            for key, p in cfgs.items():
                p = np.clip(p, EPS, 1 - EPS)
                curves.setdefault(key, np.zeros(6))[bi] = pair_brier(p, succ, tr)
        for key, cur in curves.items():
            out.setdefault(key, []).append(float(cur @ W))
    return out


def main(m):
    t0 = time.time()
    pairs = pd.read_parquet(HERE / 'pairs.parquet')
    elig = pairs.groupby('benchmark_id')['subject_id'].apply(list)
    elig = elig[elig.map(len) >= 2]
    meta = pairs.drop_duplicates('benchmark_id').set_index('benchmark_id')[['component', 'outer_fold']]
    rows = []
    for bench, subjects in elig.items():
        df = pd.read_parquet(CACHE / f'{bench}.parquet')
        df = df[df['trials'] > 0]
        subjects = sorted(set(subjects))
        items_by_subject = {s: g.set_index('item_id')[['successes', 'trials']]
                            for s, g in df[df['subject_id'].isin(subjects)].groupby('subject_id')}
        subjects = [s for s in subjects if s in items_by_subject]
        for seed in SEEDS:
            for ci, cohort in enumerate(simulate(bench, items_by_subject, subjects, m, seed)):
                sc = cohort_scores(cohort)
                for key, vals in sc.items():
                    rows.append({'benchmark_id': bench, 'seed': seed, 'cohort': ci, 'base': key[0],
                                 'lam': key[1], 'c': key[2], 'alc': float(np.mean(vals)), 'n_pairs': len(vals)})
        print(f'{bench}: done {time.time() - t0:.0f}s', flush=True)
    res = pd.DataFrame(rows)
    res = res.merge(meta, left_on='benchmark_id', right_index=True)
    res.to_parquet(HERE / f'e1_scores_m{m}.parquet', index=False)
    analyse(res, m)


def analyse(res, m):
    # benchmark x seed level means
    bs = res.groupby(['benchmark_id', 'component', 'outer_fold', 'seed', 'base', 'lam', 'c'], as_index=False)['alc'].mean()
    key = lambda d: d.base + '|' + d.lam.map(lambda v: f'{v:g}') + '|' + d.c.map(lambda v: f'{v:g}')
    bs['cfg'] = key(bs)
    wide = bs.pivot_table(index=['benchmark_id', 'component', 'outer_fold', 'seed'], columns='cfg', values='alc').reset_index()
    report = {'m': m, 'benchmarks': int(wide.benchmark_id.nunique()), 'cohort_rows': int(len(res) // len(bs.cfg.unique()))}
    for base in ('A', 'B'):
        cands = [c for c in wide.columns if isinstance(c, str) and c.startswith(base + '|') and not c.endswith('|0|0')]
        held = []; chosen = []
        for f in range(5):
            train = wide[wide.outer_fold != f]
            tb = train.groupby('benchmark_id')[cands].mean().mean()
            best = tb.idxmin(); chosen.append({'fold': f, 'cfg': best, 'train_alc': float(tb.min())})
            test = wide[wide.outer_fold == f][['benchmark_id', 'component', 'seed', 'A|0|0', 'B|0|0', best]].copy()
            test['e1'] = test[best]; held.append(test.drop(columns=[best]))
        h = pd.concat(held)
        for ctrl in ('A|0|0', 'B|0|0'):
            h['d'] = h['e1'] - h[ctrl]
            per_seed = h.groupby('seed')['d'].apply(lambda s: s.groupby(h.loc[s.index, 'benchmark_id']).mean().mean())
            bm = h.groupby(['benchmark_id', 'component'])['d'].mean().reset_index()
            comp = bm.groupby('component')['d'].agg(['sum', 'count'])
            rng = np.random.default_rng(98217); idx = rng.integers(0, len(comp), size=(10000, len(comp)))
            boot = comp['sum'].to_numpy()[idx].sum(1) / comp['count'].to_numpy()[idx].sum(1)
            report[f'E1({base}) - {ctrl}'] = {
                'mean_diff': float(bm.d.mean()), 'ci95': [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                'seed_means': {int(k): float(v) for k, v in per_seed.items()},
                'seed_sd': float(per_seed.std(ddof=1)), 'seeds_negative': int((per_seed < 0).sum()),
                'benchmarks_improved': int((bm.d < 0).sum()), 'benchmarks': int(len(bm)),
                'control_alc': float(h.groupby('benchmark_id')[ctrl].mean().mean()),
                'e1_alc': float(h.groupby('benchmark_id')['e1'].mean().mean())}
        report[f'chosen_{base}'] = chosen
    allc = wide.groupby('benchmark_id')[[c for c in wide.columns if isinstance(c, str) and '|' in c]].mean().mean()
    report['full_grid_mean_alc'] = {k: float(v) for k, v in allc.sort_values().items()}
    (HERE / f'e1_report_m{m}.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'full_grid_mean_alc'}, indent=2))


if __name__ == '__main__':
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    if len(sys.argv) > 2 and sys.argv[2] == 'analyse':
        analyse(pd.read_parquet(HERE / f'e1_scores_m{m}.parquet'), m)
    else:
        main(m)
