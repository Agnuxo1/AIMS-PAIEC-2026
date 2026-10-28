"""API parity: the shipped predict() must reproduce the simulator's E1(B|0.75|1) scores."""
import importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd
import run_e1

REPO = Path(sys.argv[1])
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
e1 = load('e1', REPO / 'submissions/item_peer/model.py')
orig = load('orig', REPO / 'submissions/original/model.py')
ctx = load('ctx', REPO / 'submissions/context_experimental/model.py')

pairs = pd.read_parquet(run_e1.HERE / 'pairs.parquet')
el = pairs.groupby('benchmark_id').subject_id.apply(lambda s: sorted(set(s)))
el = el[el.map(len) >= 2]
subj = lambda s: {k: (s if k == 'normalized_name' else '') for k in ('normalized_name', 'provider', 'release_date', 'access_date', 'harness', 'harness_version', 'reasoning_effort', 'subject_features_extra')}
item = lambda it, b: {'item_content': str(it), 'item_features': '', 'interactors': '', 'benchmark_id': b}
checked = maxerr = 0; solo_checked = 0
for bench in el.index[::6]:
    df = pd.read_parquet(run_e1.CACHE / f'{bench}.parquet'); df = df[df.trials > 0]
    ibs = {s: g.set_index('item_id')[['successes', 'trials']] for s, g in df[df.subject_id.isin(el[bench])].groupby('subject_id')}
    for m in (2, 4):
        for cohort in run_e1.simulate(bench, ibs, [s for s in el[bench] if s in ibs], m, 17)[:1]:
            ref = run_e1.cohort_scores(cohort)
            for key, model in ((('B', 0.75, 1), e1), (('B', 0, 0), ctx)):
                got = []
                for j, P in enumerate(cohort):
                    curve = np.zeros(6)
                    for bi, n in enumerate(run_e1.N):
                        labeled = [[[subj(Q['subject']), item(it, bench)], int(y)] for Q in cohort for it, y in zip(Q['acq_items'][:n], Q['y'][:n])]
                        p = np.array([model.predict([subj(P['subject']), item(it, bench)], labeled) for it in P['ev_items']])
                        curve[bi] = run_e1.pair_brier(p, P['ev_succ'], P['ev_trials'])
                    got.append(curve @ run_e1.W)
                err = float(np.max(np.abs(np.array(got) - np.array(ref[key]))))
                maxerr = max(maxerr, err); checked += len(got)
            # fallback: a single visible subject must match the submitted predictor exactly
            P = cohort[0]
            for n in run_e1.N:
                labeled = [[[subj(P['subject']), item(it, bench)], int(y)] for it, y in zip(P['acq_items'][:n], P['y'][:n])]
                for it in P['ev_items'][:10]:
                    a = e1.predict([subj(P['subject']), item(it, bench)], labeled); b = orig.predict([subj(P['subject']), item(it, bench)], labeled)
                    assert a == b, (a, b); solo_checked += 1
print({'pair_curves_checked': checked, 'max_abs_alc_error': maxerr, 'fallback_equal_checks': solo_checked})
assert maxerr < 1e-12
