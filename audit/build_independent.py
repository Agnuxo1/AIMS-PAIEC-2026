"""Independent raw-data reconstruction; no imports from the previous project.

Exact sufficient statistics for predictors constant within a subject-benchmark
pair: E[(p-Y)^2] = p*p - 2*p*q + q, q = sum(Y)/number of responses.
Each recorded evaluation response contributes, without materializing predictions.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='4'
import gc, hashlib, json, sqlite3, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parent
DATA=Path(os.environ.get('AIMS_DATA',r'E:\AIMS-sprint\data'))
SEEDS=(17,29,43,71,101,137,181,233)
BUDGETS=np.array([0,1,3,7,15,31],dtype=np.int64)
WEIGHTS=np.array([.1,.2,.2,.2,.2,.1])
CACHE_SEMANTICS='count-sum-first-v1'

def stable(*parts):
    b=json.dumps(parts,ensure_ascii=True,separators=(',',':')).encode()
    return int.from_bytes(hashlib.sha256(b).digest()[:8],'little')

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def raw_aggregate(path):
    pieces=[]; values=set(); n_null=0; n_rows=0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=250_000,columns=['subject_id','item_id','response']):
        t=pa.Table.from_batches([batch]); n_rows+=len(t)
        n_null+=t['response'].null_count
        vals=pc.unique(t['response']).to_pylist()
        values.update(v for v in vals if v is not None)
        if not values.issubset({0.,1.}):
            return None,{'rows':n_rows,'nulls':n_null,'observed_support_is_binary':False,
                         'values_preview':sorted(str(v) for v in values)[:12],'scan':'nonbinary witness; early stop'}
        t=t.filter(pc.is_valid(t['response']))
        if not len(t):continue
        assert t['subject_id'].null_count==0 and t['item_id'].null_count==0
        g=t.group_by(['subject_id','item_id'],use_threads=False).aggregate([
            ('response','sum'),('response','count'),('response','first')])
        pieces.append(g)
    info={'rows':n_rows,'nulls':n_null,'observed_support_is_binary':bool(values),
          'values_preview':sorted(values),'scan':'complete'}
    if not pieces:return None,info
    t=pa.concat_tables(pieces).group_by(['subject_id','item_id'],use_threads=False).aggregate([
        ('response_sum','sum'),('response_count','sum'),('response_first','first')])
    t=t.rename_columns(['subject_id','item_id','successes','trials','first'])
    return t,info

def summarize_pair(g,bench,sid,attributes):
    g=g.sort_values('item_id',kind='stable'); n=len(g)
    assert n>=80
    counts=g['trials'].to_numpy(np.int64); sums=g['successes'].to_numpy(np.int64)
    first=g['first'].to_numpy(np.int8); means=sums/counts
    row={'benchmark_id':bench,'subject_id':str(sid),'n_items':n,'n_responses':int(counts.sum()),
         'mean_response':float(sums.sum()/counts.sum()),'mean_item':float(means.mean()),
         'mixed_items':int(((sums>0)&(sums<counts)).sum()),'repeated_items':int((counts>1).sum())}
    row.update(attributes)
    scenarios=[]
    for seed in SEEDS:
        rng=np.random.default_rng(stable('AIMS-independent-v1',seed,bench,str(sid)))
        perm=rng.permutation(n); acq,ev=perm[:n//2],perm[n//2:]
        assert len(set(acq[:31]).intersection(ev))==0 and len(acq)>=31
        acquired=acq[:31]
        y=(rng.random(31)<means[acquired]).astype(np.int8)
        cumulative=np.r_[0,np.cumsum(y)]; first_cum=np.r_[0,np.cumsum(first[acquired])]
        q=float(sums[ev].sum()/counts[ev].sum())
        sc={'benchmark_id':bench,'subject_id':str(sid),'seed':seed,'q_eval':q,
            'n_eval_responses':int(counts[ev].sum()),'n_eval_items':len(ev),
            'q_eval_items':float(means[ev].mean()),
            'oracle_item_brier':float(np.sum(sums[ev]*(1-means[ev]))/counts[ev].sum())}
        sc.update({f'k{n}':int(cumulative[n]) for n in BUDGETS})
        sc.update({f'first_k{n}':int(first_cum[n]) for n in BUDGETS})
        scenarios.append(sc)
    return row,scenarios

def main():
    pa.set_cpu_count(4); pa.set_io_thread_count(2)
    t0=time.time(); cache=ROOT/'raw_cache'; cache.mkdir(exist_ok=True)
    metas=[]; inventory=[]; pair_rows=[]; scenario_rows=[]; parent={}; overlap=Counter()
    database=ROOT/'hashes.sqlite'
    db=sqlite3.connect(database)
    db.execute('PRAGMA cache_size=-16000')
    db.execute('DROP TABLE IF EXISTS hashes')
    db.execute('CREATE TABLE hashes (h TEXT NOT NULL, b TEXT NOT NULL)')
    source_manifest=[]
    def find(x):
        parent.setdefault(x,x)
        while parent[x]!=x:parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    for path in sorted(DATA.glob('*/formatted_tables/benchmarks.parquet')):
        ms=pq.read_table(path).to_pylist()
        assert len(ms)==1, str(path)
        m=ms[0]; bench=m['benchmark_id']; metas.append(m)
        resp=path.with_name('responses.parquet')
        entry={'benchmark_id':bench,'release':m['release'],'declared_type':m['response_type'],
               'declared_n_values':m['n_response_values'],'eligible_pairs':0}
        if m['release']!='public' or not resp.exists():
            entry['status']='not_public_or_missing';inventory.append(entry);continue
        sha=digest(resp); meta_cache=cache/f'{bench}.json'; agg_cache=cache/f'{bench}.parquet'
        if (meta_cache.exists() and json.loads(meta_cache.read_text()).get('cache_semantics')==CACHE_SEMANTICS
                and json.loads(meta_cache.read_text())['source_sha256']==sha
                and (not json.loads(meta_cache.read_text())['observed_support_is_binary'] or agg_cache.exists())):
            info=json.loads(meta_cache.read_text()); table=pq.read_table(agg_cache) if agg_cache.exists() and info['observed_support_is_binary'] else None
        else:
            table,info=raw_aggregate(resp);info['source_sha256']=sha;info['cache_semantics']=CACHE_SEMANTICS
            if table is not None:pq.write_table(table,agg_cache,compression='zstd')
            meta_cache.write_text(json.dumps(info,indent=2),encoding='utf-8')
        entry.update(info)
        hf=DATA/'.cache/huggingface/download'/bench/'formatted_tables/responses.parquet.metadata'
        if hf.exists():
            ls=hf.read_text().splitlines();entry['hf_commit']=ls[0];entry['hf_etag']=ls[1]
            entry['sha_matches_hf_etag']=sha==ls[1]
        source_manifest.append({'benchmark_id':bench,'sha256':sha,'bytes':resp.stat().st_size,
                                'commit':entry.get('hf_commit'),'etag_matches':entry.get('sha_matches_hf_etag')})
        if table is None or not info['observed_support_is_binary']:
            entry['status']='nonbinary_or_empty';inventory.append(entry);continue
        subject_path=path.with_name('subjects.parquet')
        subjects=pq.read_table(subject_path).to_pandas().set_index('subject_id') if subject_path.exists() else pd.DataFrame()
        d=table.to_pandas();eligible=d.groupby('subject_id',sort=True).size();eligible=eligible[eligible>=80].index
        for sid,g in d[d['subject_id'].isin(eligible)].groupby('subject_id',sort=True):
            a={}
            if sid in subjects.index:
                sub=subjects.loc[sid]
                assert isinstance(sub,pd.Series),'duplicate subject metadata'
                for key in ['normalized_name','provider','release_date','access_date','harness','harness_version','reasoning_effort','subject_features_extra']:
                    value=sub.get(key);a[key]='' if pd.isna(value) else str(value)
            row,sc=summarize_pair(g,bench,sid,a);pair_rows.append(row);scenario_rows.extend(sc)
        entry['eligible_pairs']=len(eligible);entry['status']='included' if len(eligible) else 'binary_no_eligible_pair'
        item_path=path.with_name('items.parquet')
        if item_path.exists():
            hash_col=pq.read_table(item_path,columns=['content_hash'])['content_hash']
            hset=set(hash_col.to_pylist());hset.discard(None);hset.discard('');entry['unique_content_hashes']=len(hset)
            db.executemany('INSERT INTO hashes VALUES (?,?)',((h,bench) for h in hset))
            db.commit()
        parent.setdefault(bench,bench)
        inventory.append(entry)
        if len(inventory)%10==0:print(f'{len(inventory)}/{len(list(DATA.glob("*/formatted_tables/benchmarks.parquet")))} sources; {len(pair_rows)} eligible pairs; {time.time()-t0:.1f}s',flush=True)
        del d,table,subjects
        gc.collect()
    print('Building on-disk cross-benchmark hash index',flush=True)
    db.execute('CREATE INDEX hash_idx ON hashes(h)');db.commit()
    shared_hashes=0
    for h,benches in db.execute('SELECT h, GROUP_CONCAT(b) FROM hashes GROUP BY h HAVING COUNT(*) > 1'):
        bs=sorted(set(benches.split(',')));shared_hashes+=1
        for b in bs[1:]:union(bs[0],b)
        for i,a in enumerate(bs):
            for b in bs[i+1:]:overlap[a+'|'+b]+=1
    distinct_hashes=db.execute('SELECT COUNT(DISTINCT h) FROM hashes').fetchone()[0]
    db.close()
    pairs=pd.DataFrame(pair_rows);scenarios=pd.DataFrame(scenario_rows)
    components={b:find(b) for b in pairs['benchmark_id'].unique()}
    group_members={g:sorted(b for b in components if components[b]==g) for g in set(components.values())}
    sorted_groups=sorted(group_members,key=lambda g:stable('outer-fold',g))
    folds={g:i%5 for i,g in enumerate(sorted_groups)}
    pairs['component']=pairs['benchmark_id'].map(components);pairs['outer_fold']=pairs['component'].map(folds)
    scenarios=scenarios.merge(pairs[['benchmark_id','subject_id','component','outer_fold']],on=['benchmark_id','subject_id'],validate='many_to_one')
    pairs.to_parquet(ROOT/'pairs.parquet',index=False);scenarios.to_parquet(ROOT/'scenarios.parquet',index=False)
    pd.DataFrame(inventory).to_json(ROOT/'inventory.json',orient='records',indent=2)
    (ROOT/'source_manifest.json').write_text(json.dumps(source_manifest,indent=2),encoding='utf-8')
    (ROOT/'fold_manifest.json').write_text(json.dumps({'groups':group_members,'folds':folds,'overlap_counts':dict(overlap)},indent=2),encoding='utf-8')
    catalog=DATA.parent/'meta/catalogo_benchmarks.parquet'
    cat=pd.read_parquet(catalog) if catalog.exists() else pd.DataFrame(metas)
    old=set(cat.loc[(cat.release=='public')&(cat.n_response_values==2),'benchmark_id'])
    current=set(pairs.benchmark_id)
    summary={'benchmarks_in_metadata':len(metas),'included_benchmarks':len(current),'eligible_pairs':len(pairs),
             'scenarios':len(scenarios),'components':len(group_members),'new_benchmarks_vs_old_filter':sorted(current-old),
             'shared_content_hashes':shared_hashes,'distinct_content_hashes':distinct_hashes,
             'full_eligible_pair_weights':pairs.benchmark_id.value_counts().head(10).to_dict(),
             'cache_revisions':dict(Counter(x.get('commit') for x in source_manifest)),
             'sha_etag_mismatches':[x['benchmark_id'] for x in source_manifest if x['etag_matches'] is False],
             'binary_rows_including_missing':sum(x['rows'] for x in inventory if x.get('observed_support_is_binary')),
             'binary_nonmissing_responses':sum(x['rows']-x['nulls'] for x in inventory if x.get('observed_support_is_binary')),
             'elapsed_seconds':time.time()-t0,'protocol_sha256':digest(ROOT/'PROTOCOL.md'),'builder_sha256':digest(Path(__file__))}
    (ROOT/'build_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
