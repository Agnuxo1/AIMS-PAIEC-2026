"""Exploratory peer-label experiment with explicit availability sensitivities."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='4'
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from evaluate_independent import N,W,ROOT,stable,alc

def cohort_evidence(pairs,d,k,size):
    ownk=np.zeros_like(k);ownn=np.zeros_like(k);peerk=np.zeros_like(k);peern=np.zeros_like(k)
    cols=['normalized_name','provider','release_date','access_date','harness','harness_version','reasoning_effort','subject_features_extra']
    signatures=pairs[cols].fillna('').astype(str).apply(lambda row:json.dumps(row.to_dict(),sort_keys=True),axis=1).to_dict()
    for (bench,seed),g in d.groupby(['benchmark_id','seed'],sort=True):
        ids=g.index.to_numpy();rng=np.random.default_rng(stable('peer-cohort',bench,int(seed)))
        shuffled=rng.permutation(ids)
        for start in range(0,len(shuffled),size):
            group=shuffled[start:start+size];total=k[group].sum(axis=0)
            sig=np.array([signatures[int(d.at[i,'pair_index'])] for i in group])
            for signature in set(sig):
                same=group[sig==signature];samek=k[same].sum(axis=0)
                ownk[same]=samek;ownn[same]=len(same)*N
                peerk[same]=total-samek;peern[same]=(len(group)-len(same))*N
    return ownk,ownn,peerk,peern

def prediction(evidence,mix,strength):
    ok,on,pk,pn=evidence
    peer=(1.5+pk)/(3.+pn);m=.5+mix*(peer-.5)
    return (ok+strength*m)/(on+strength)

def main():
    pairs=pd.read_parquet(ROOT/'pairs.parquet').sort_values(['benchmark_id','subject_id']).reset_index(drop=True)
    pairs['pair_index']=np.arange(len(pairs))
    d=pd.read_parquet(ROOT/'scenarios.parquet').merge(pairs[['benchmark_id','subject_id','pair_index']],on=['benchmark_id','subject_id'],validate='many_to_one')
    d=d.sort_values(['pair_index','seed']).reset_index(drop=True)
    k=d[[f'k{n}' for n in N]].to_numpy(np.int64);q=d.q_eval.to_numpy()
    ev={size:cohort_evidence(pairs,d,k,size) for size in (1,2,4,8)}
    np.testing.assert_array_equal(prediction(ev[1],0,3),prediction(ev[1],1,3))
    scores=d[['benchmark_id','subject_id','seed','component','outer_fold']].copy()
    chosen=[];logs=[]
    for size in ev:
        scores[f'original_cohort{size}']=alc(prediction(ev[size],0,3),q)
        scores[f'context_cohort{size}']=np.nan
        scores[f'fallback_safe_cohort{size}']=np.nan
    for fold in range(5):
        external=d.outer_fold.to_numpy()==fold
        roles=json.loads((ROOT/f'fold-{fold}-roles.json').read_text())
        validation=d.benchmark_id.isin(roles['validation']).to_numpy()
        candidates=[];safe=[]
        for mix in (0,.25,.5,1):
            for strength in (1,3,8,16):
                val=float(alc(prediction(tuple(x[validation] for x in ev[4]),mix,strength),q[validation]).mean())
                cfg={'mix':mix,'strength':strength};candidates.append((val,cfg))
                if strength==3:safe.append((val,cfg))
                logs.append({'fold':fold,'validation_alc':val,**cfg})
        for family,candidate in [('context',candidates),('fallback_safe',safe)]:
            val,cfg=min(candidate,key=lambda x:x[0]);chosen.append({'fold':fold,'family':family,'validation_alc':val,**cfg})
            for size in ev:
                pred=prediction(tuple(x[external] for x in ev[size]),**cfg)
                scores.loc[external,f'{family}_cohort{size}']=alc(pred,q[external])
        print('context fold',fold,'complete',flush=True)
    # BLAS reduction across differently sized batches can differ by one ULP;
    # probability-level fallback equality above is exact, score equality is numeric.
    assert np.allclose(scores.original_cohort1,scores.fallback_safe_cohort1,atol=1e-14,rtol=0)
    # Canonicalize a mathematically identical control; otherwise one-ULP BLAS
    # differences could be misreported as a statistically nonzero effect.
    scores['fallback_safe_cohort1']=scores['original_cohort1']
    pd.DataFrame(logs).to_csv(ROOT/'context_all_validation_candidates.csv',index=False)
    (ROOT/'context_selected.json').write_text(json.dumps(chosen,indent=2),encoding='utf-8')
    scores.to_parquet(ROOT/'context_scores.parquet',index=False)
    means=scores.groupby(['benchmark_id','subject_id','component'],as_index=False).mean(numeric_only=True)
    components=sorted(means.component.unique());rng=np.random.default_rng(98217)
    idx=rng.integers(0,len(components),size=(10000,len(components)))
    summaries=[]
    for size in ev:
        ref=f'original_cohort{size}'
        for family in ('context','fallback_safe'):
            name=f'{family}_cohort{size}';m=means.assign(diff=means[name]-means[ref]);g=m.groupby('component')['diff'].agg(['sum','count']).reindex(components)
            boot=g['sum'].to_numpy()[idx].sum(axis=1)/g['count'].to_numpy()[idx].sum(axis=1)
            row={'method':name,'reference':ref,'brier_alc':float(scores[name].mean()),'reference_alc':float(scores[ref].mean()),
                 'difference':float(m['diff'].mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),
                 'ci9875_four_families':np.quantile(boot,[.05/8,1-.05/8]).tolist(),
                 'ci99_five_candidates':np.quantile(boot,[.05/10,1-.05/10]).tolist(),
                 'benchmark_mean_difference':float(m.groupby('benchmark_id')['diff'].mean().mean()),
                 'difference_by_seed':scores.assign(diff=scores[name]-scores[ref]).groupby('seed')['diff'].mean().to_dict()}
            summaries.append(row)
    (ROOT/'context_summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    print(json.dumps(summaries,indent=2),flush=True)

if __name__=='__main__':main()
