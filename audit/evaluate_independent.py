"""Nested grouped evaluation, written independently of the original harness."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='4'
import hashlib, importlib.util, json, re, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge

ROOT=Path(__file__).resolve().parent
N=np.array([0,1,3,7,15,31]);W=np.array([.1,.2,.2,.2,.2,.1])

def stable(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts,separators=(',',':')).encode()).digest()[:8],'little')

def brier(p,q):
    assert np.isfinite(p).all() and np.logical_and(p>=0,p<=1).all()
    return p*p-2*p*q[:,None]+q[:,None]

def alc(p,q):return brier(p,q)@W

def beta(k,mean=.5,strength=3):
    m=np.asarray(mean)
    if m.ndim:m=m[:,None]
    return np.clip((k+strength*m)/(N+strength),1e-6,1-1e-6)

def calibrated_table(k,q):
    tables=[]
    for j,n in enumerate(N):
        counts=np.bincount(k[:,j],minlength=n+1)
        sums=np.bincount(k[:,j],weights=q,minlength=n+1)
        base=(np.arange(n+1)+1.5)/(n+3.)
        tables.append((sums+16*base)/(counts+16))
    return tables

def apply_table(tables,k):
    return np.column_stack([tables[j][k[:,j]] for j in range(6)])

def subject_features(pair):
    out=[]
    for row in pair.fillna('').to_dict('records'):
        d={}
        for field in ('normalized_name','provider','harness','harness_version','reasoning_effort'):
            v=str(row.get(field,'')).strip().lower()
            if v:d[field+'='+v]=1.
        for tok in set(re.findall(r'[a-z]+|\d+',str(row.get('normalized_name','')).lower())):
            d['name_token='+tok]=1.
        date=str(row.get('release_date',''))
        year=re.search(r'20\d\d',date)
        d['release_year']=float(np.clip(int(year.group())-2024,-8,4)) if year else 0.
        d['release_missing']=float(not bool(year))
        out.append(d)
    return out

def conformity_tests():
    rng=np.random.default_rng(923)
    # Algebraic sufficient statistics vs explicit independent per-response MSE.
    for _ in range(150):
        y=rng.integers(0,2,size=int(rng.integers(1,500)))
        p=rng.random(6)
        direct=np.array([np.mean((x-y)**2) for x in p])
        assert np.allclose(direct,brier(p[None,:],np.array([y.mean()]))[0],atol=1e-14)
        assert abs(np.trapezoid(direct,dx=1)/5-direct@W)<1e-14
    # The delivered code is the object under test, never the expected-value oracle.
    delivered=Path(os.environ.get('AIMS_ORIGINAL_MODEL',str(ROOT.parent/'submissions/original/model.py')))
    if not delivered.exists():delivered=Path(r'E:\AIMS-sprint\entrega\repo\submission\model.py')
    spec=importlib.util.spec_from_file_location('audited_submission',delivered)
    model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)
    subject={key:'' for key in ('normalized_name','provider','release_date','access_date','harness','harness_version','reasoning_effort','subject_features_extra')}
    subject['normalized_name']='synthetic-subject'
    for n in N:
        for k in range(n+1):
            item={'benchmark_id':'anon-A','item_content':'target','item_features':'','interactors':''}
            evidence=[[[subject,dict(item,item_content=f'evidence-{i}')],int(i<k)] for i in range(n)]
            expected=(k+1.5)/(n+3)
            observed=model.predict([subject,item],evidence)
            assert abs(observed-expected)<1e-14
            noise=[[[dict(subject,normalized_name='other'),item],1],[[subject,dict(item,benchmark_id='anon-B')],0]]
            assert model.predict([subject,item],evidence+noise)==observed
            rng.shuffle(evidence)
            assert model.predict([subject,item],evidence)==observed
            assert model.predict([subject,dict(item,item_content='different text')],evidence)==observed
    # A planted signal must be detected by the learning family (positive control).
    k=np.zeros((400,6),dtype=int);q=np.r_[np.zeros(200),np.ones(200)]
    k[200:,:]=N
    table=calibrated_table(k,q)
    assert apply_table(table,k)[:200,1].mean()<.1
    assert apply_table(table,k)[200:,1].mean()>.9
    return {'independent_metric_cases':150,'delivered_all_valid_n_k':int(sum(n+1 for n in N)),
            'invariants':['evidence order','other subject ignored','other benchmark ignored','target text invariance'],
            'positive_control':'calibration learns planted association','status':'pass'}

def nested_evaluation():
    pairs=pd.read_parquet(ROOT/'pairs.parquet').sort_values(['benchmark_id','subject_id']).reset_index(drop=True)
    pairs['pair_index']=np.arange(len(pairs))
    d=pd.read_parquet(ROOT/'scenarios.parquet').merge(pairs[['benchmark_id','subject_id','pair_index']],on=['benchmark_id','subject_id'],validate='many_to_one')
    d=d.sort_values(['pair_index','seed']).reset_index(drop=True)
    k=d[[f'k{n}' for n in N]].to_numpy(np.int64);q=d.q_eval.to_numpy()
    base=beta(k)
    empirical=np.divide(k,N,out=np.full(k.shape,.5,dtype=float),where=N>0)
    first=d[[f'first_k{n}' for n in N]].to_numpy(np.int64)
    predictions={'constant_05':np.full_like(base,.5),'empirical_mean':empirical,'original_beta3':base,
                 'tuned_beta':np.zeros_like(base),'calibration_table':np.zeros_like(base),'subject_ridge':np.zeros_like(base),
                 'original_beta3_first_repeat':beta(first)}
    features=subject_features(pairs)
    logs=[];selected=[]
    for fold in range(5):
        external=(d.outer_fold.to_numpy()==fold);training=~external
        groups=sorted(d.loc[training,'component'].unique(),key=lambda g:stable('inner-validation',fold,g))
        valgroups=set(groups[::5]);validation=training & d.component.isin(valgroups).to_numpy();fit=training&~validation
        assert not (set(d.loc[fit,'component']) & set(d.loc[validation|external,'component']))
        assert not (set(d.loc[validation,'component']) & set(d.loc[external,'component']))
        masks={'fit':fit,'validation':validation,'test':external}
        manifest={key:sorted(d.loc[mask,'benchmark_id'].unique()) for key,mask in masks.items()}
        (ROOT/f'fold-{fold}-roles.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        # All 21 beta candidates are logged; validation alone selects one.
        candidates=[]
        for mean in (.4,.5,.6):
            for strength in (.5,1,2,3,5,8,16):
                score=float(alc(beta(k[validation],mean,strength),q[validation]).mean())
                params={'mean':mean,'strength':strength}
                candidates.append((score,params));logs.append({'fold':fold,'family':'tuned_beta','validation_alc':score,**params})
        score,params=min(candidates,key=lambda x:x[0])
        predictions['tuned_beta'][external]=beta(k[external],**params)
        selected.append({'fold':fold,'family':'tuned_beta','validation_alc':score,**params})
        # Calibration fit is isolated from validation and external outcomes.
        tables=calibrated_table(k[fit],q[fit]);tableval=apply_table(tables,k[validation]);candidates=[]
        for mix in (0,.25,.5,1):
            val=(1-mix)*base[validation]+mix*tableval
            score=float(alc(val,q[validation]).mean())
            candidates.append((score,mix));logs.append({'fold':fold,'family':'calibration_table','validation_alc':score,'mix':mix})
        score,mix=min(candidates,key=lambda x:x[0])
        predictions['calibration_table'][external]=(1-mix)*base[external]+mix*apply_table(tables,k[external])
        selected.append({'fold':fold,'family':'calibration_table','validation_alc':score,'mix':mix})
        # Train only on fit pairs; eight seeds never count as eight independent subjects.
        fitids=np.sort(d.loc[fit,'pair_index'].unique());validids=np.sort(d.loc[validation,'pair_index'].unique());testids=np.sort(d.loc[external,'pair_index'].unique())
        pair_target=d.loc[fit].groupby('pair_index').q_eval.mean().reindex(fitids).to_numpy()
        vec=DictVectorizer(sparse=True);Xfit=vec.fit_transform([features[i] for i in fitids])
        Xval=vec.transform([features[i] for i in validids]);Xtest=vec.transform([features[i] for i in testids])
        candidates=[];models={}
        for alpha in (1,10,100):
            model=Ridge(alpha=alpha,solver='lsqr',tol=1e-7).fit(Xfit,pair_target)
            models[alpha]=model
            prior=pd.Series(np.clip(model.predict(Xval),.01,.99),index=validids)
            v=prior.reindex(d.loc[validation,'pair_index']).to_numpy()
            for mix in (0,.25,.5,1):
                p0=.5+mix*(v-.5)
                for strength in (1,3,8):
                    score=float(alc(beta(k[validation],p0,strength),q[validation]).mean())
                    params={'alpha':alpha,'mix':mix,'strength':strength}
                    candidates.append((score,params));logs.append({'fold':fold,'family':'subject_ridge','validation_alc':score,**params})
        score,params=min(candidates,key=lambda x:x[0]);model=models[params['alpha']]
        prior=pd.Series(np.clip(model.predict(Xtest),.01,.99),index=testids).reindex(d.loc[external,'pair_index']).to_numpy()
        predictions['subject_ridge'][external]=beta(k[external],.5+params['mix']*(prior-.5),params['strength'])
        selected.append({'fold':fold,'family':'subject_ridge','validation_alc':score,**params})
        print(f'fold {fold}: fit={len(fitids)} val={len(validids)} test={len(testids)}; finished',flush=True)
    result=d[['benchmark_id','subject_id','seed','component','outer_fold','pair_index']].copy()
    summaries=[]
    for name,p in predictions.items():
        scores=brier(p,q);result[name]=scores@W
        summaries.append({'method':name,'brier_alc':float(result[name].mean()),
                          **{f'B{n}':float(scores[:,j].mean()) for j,n in enumerate(N)}})
    result.to_parquet(ROOT/'scores.parquet',index=False)
    # Keep predictions for forensic recomputation without refitting.
    np.savez_compressed(ROOT/'predictions.npz',**predictions,k=k,q=q)
    pd.DataFrame(logs).to_csv(ROOT/'all_validation_candidates.csv',index=False)
    (ROOT/'selected_hyperparameters.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
    pd.DataFrame(summaries).to_csv(ROOT/'summary.csv',index=False)
    return result,pd.DataFrame(summaries)

def uncertainty(result,summary):
    methods=[c for c in summary.method if c not in ['constant_05','empirical_mean','original_beta3','original_beta3_first_repeat']]
    means=result.groupby(['benchmark_id','subject_id','component','outer_fold'],as_index=False)[list(summary.method)].mean()
    rng=np.random.default_rng(7543);components=sorted(means.component.unique())
    indices=rng.integers(0,len(components),size=(10000,len(components)))
    rows=[]
    for name,reference in [(m,'original_beta3') for m in methods]+[('original_beta3','empirical_mean'),('original_beta3_first_repeat','original_beta3')]:
        values=means.assign(difference=means[name]-means[reference])
        g=values.groupby('component').difference.agg(['sum','count']).reindex(components)
        sums=g['sum'].to_numpy();counts=g['count'].to_numpy()
        boot=sums[indices].sum(axis=1)/counts[indices].sum(axis=1)
        ci95=np.quantile(boot,[.025,.975]);ci_adjusted=np.quantile(boot,[.05/(2*3),1-.05/(2*3)])
        delta=float(values.difference.mean())
        signs=rng.choice([-1,1],size=(10000,len(components)))
        perm=(signs*sums).sum(axis=1)/counts.sum()
        p=(1+np.count_nonzero(np.abs(perm)>=abs(delta)))/10001
        bybench=values.groupby('benchmark_id').difference.mean()
        byseed=result.assign(diff=result[name]-result[reference]).groupby('seed')['diff'].mean()
        folds=values.groupby('outer_fold').difference.mean()
        largest=g['count'].idxmax();removed=values[values.component!=largest]
        row={'method':name,'reference':reference,'difference_pair_mean':delta,'ci95':ci95.tolist(),
             'ci98333_three_families':ci_adjusted.tolist(),'cluster_sign_flip_p_plus_one':p,
             'difference_benchmark_mean':float(bybench.mean()),'benchmarks_improved':int((bybench<0).sum()),
             'benchmarks_tied':int((bybench==0).sum()),'n_benchmarks':len(bybench),'n_pairs':len(means),'n_components':len(components),
             'difference_per_seed':{str(k):float(v) for k,v in byseed.items()},'difference_per_fold':{str(k):float(v) for k,v in folds.items()},
             'largest_component':largest,'largest_component_pair_count':int(g.loc[largest,'count']),
             'difference_excluding_largest_component':float(removed.difference.mean()),
             'promote':bool(ci_adjusted[1]<0 and (byseed<0).sum()>=6)}
        rows.append(row)
    (ROOT/'uncertainty.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    means.to_parquet(ROOT/'pair_mean_scores.parquet',index=False)
    return rows

def main():
    t0=time.time();tests=conformity_tests()
    (ROOT/'independent_tests.json').write_text(json.dumps(tests,indent=2),encoding='utf-8')
    result,summary=nested_evaluation();ci=uncertainty(result,summary)
    print(summary.to_string(index=False),flush=True)
    print(json.dumps(ci,indent=2),flush=True)
    print('elapsed_seconds',time.time()-t0,flush=True)

if __name__=='__main__':main()
