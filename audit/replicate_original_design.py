"""Re-derive the original sampling design from independently aggregated raw data.

This is a replication of a known design, separate from the independent primary
evaluation. No original modules, tests or pickle objects are executed/imported.
"""
from pathlib import Path
import hashlib,json,os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parent
ORIGINAL=Path(os.environ.get('AIMS_LEGACY_ROOT',r'E:\AIMS-sprint'))
N=np.array([0,1,3,7,15,31]);W=np.array([.1,.2,.2,.2,.2,.1]);SEED=20261024

def original_pair_seed(sid,bid):
    h=hashlib.blake2b(digest_size=8)
    for value in (SEED,sid,bid):h.update(str(value).encode());h.update(b'|')
    return int.from_bytes(h.digest(),'big')%(2**32)

def raw_item_order(bench,sids):
    # Arrow hash-group output order is not a contractual first-appearance order.
    # Independently scan raw keys to recover the ordering used by pandas.unique.
    orders={sid:{} for sid in sids};valid=pa.array(sids)
    path=ORIGINAL/'data'/bench/'formatted_tables/responses.parquet'
    for batch in pq.ParquetFile(path).iter_batches(batch_size=250000,columns=['subject_id','item_id','response']):
        t=pa.Table.from_batches([batch]);mask=pc.and_(pc.is_in(t['subject_id'],value_set=valid),pc.is_valid(t['response']))
        t=t.filter(mask).select(['subject_id','item_id'])
        f=t.to_pandas().drop_duplicates(['subject_id','item_id'])
        for sid,item in f.itertuples(index=False,name=None):orders[sid].setdefault(item,None)
    return orders

def main():
    cat=pd.read_parquet(ORIGINAL/'meta/catalogo_benchmarks.parquet')
    benches=sorted(cat.loc[(cat.release=='public')&(cat.n_response_values==2),'benchmark_id'])
    crop_rng=np.random.default_rng(SEED);rows=[]
    for bench in benches:
        path=ROOT/'raw_cache'/f'{bench}.parquet'
        if not path.exists():continue
        d=pd.read_parquet(path)
        eligible=d.groupby('subject_id',sort=True).size();sids=eligible[eligible>=80].index.to_list()
        if len(sids)>6:
            choice=np.sort(np.random.default_rng(SEED).choice(len(sids),6,replace=False))
            sids=[sids[i] for i in choice]
        if not sids:continue
        orders=raw_item_order(bench,sids)
        for sid in sids:
            g=d[d.subject_id==sid].set_index('item_id').reindex(list(orders[sid])).reset_index()
            assert not g.trials.isna().any()
            size=len(g);perm=np.random.default_rng(original_pair_seed(sid,bench)).permutation(size)
            acquired=perm[:size//2];selected_eval=np.sort(perm[size//2:])
            if len(selected_eval)>150:
                selected_eval=selected_eval[np.sort(crop_rng.choice(len(selected_eval),150,replace=False))]
            count=g.trials.to_numpy();success=g.successes.to_numpy();first=g['first'].to_numpy()
            q=float(success[selected_eval].sum()/count[selected_eval].sum());k=np.r_[0,np.cumsum(first[acquired[:31]])][N]
            beta=(1.5+k)/(3+N);empirical=np.divide(k,N,out=np.full(6,.5),where=N>0)
            rows.append({'benchmark_id':bench,'subject_id':sid,'beta':float((beta*beta-2*beta*q+q)@W),
                         'empirical':float((empirical*empirical-2*empirical*q+q)@W)})
    frame=pd.DataFrame(rows)
    original=pd.read_parquet(ORIGINAL/'work/res_beta_binomial_p005.parquet')
    official=pd.read_parquet(ORIGINAL/'work/res_media_empirica_OFICIAL.parquet')
    merged=original.merge(frame,on=['benchmark_id','subject_id'],validate='one_to_one')
    b=official.merge(frame,on=['benchmark_id','subject_id'],validate='one_to_one')
    summary={'reconstructed_pairs_all':len(frame),'matched_headline_pairs':len(merged),
             'original_headline_pairs':len(original),'max_beta_pair_error':float(np.max(np.abs(merged.brier_alc-merged.beta))),
             'max_empirical_pair_error':float(np.max(np.abs(b.brier_alc-b.empirical))),
             'reconstructed_beta_mean':float(merged.beta.mean()),'reconstructed_empirical_mean':float(b.empirical.mean())}
    summary['exact_to_numeric_tolerance']=bool(len(merged)==len(original) and summary['max_beta_pair_error']<1e-12 and summary['max_empirical_pair_error']<1e-12)
    frame.to_parquet(ROOT/'replicated_original_pairs.parquet',index=False)
    (ROOT/'original_raw_replication.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
