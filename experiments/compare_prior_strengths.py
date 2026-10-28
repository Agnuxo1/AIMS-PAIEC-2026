import argparse,json,hashlib,datetime
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=ROOT/'results/scenarios.parquet');args=ap.parse_args()
 d=pd.read_parquet(args.source); n=np.array([0,1,3,7,15,31]);w=np.array([.1,.2,.2,.2,.2,.1]);k=d[[f'k{x}' for x in n]].to_numpy();q=d.q_eval.to_numpy()[:,None]
 assert len(d)==105600 and not d.duplicated(['benchmark_id','subject_id','seed']).any()
 def loss(s):
  p=(k+.5*s)/(n+s)
  assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
  return (p*p-2*p*q+q)@w
 vals={'baseline_S3':loss(3),'A1_S5':loss(5),'A2_S1':loss(1),'A3_increasing_strength':loss(3+np.sqrt(n))}
 a=d[['benchmark_id','subject_id','component','seed']].copy()
 for name,v in vals.items():a[name]=v
 pairs=a.groupby(['benchmark_id','subject_id','component'],as_index=False)[list(vals)].mean()
 components=pairs.component.unique();rng=np.random.default_rng(20261028)
 ix={g:np.flatnonzero(pairs.component.to_numpy()==g) for g in components}
 samples=[np.concatenate([ix[g] for g in rng.choice(components,len(components),replace=True)]) for _ in range(2000)]
 out={'kind':'new_exploratory_shrinkage_interventions_historical_baseline','n_scenarios':len(d),'n_pairs':len(pairs),'n_benchmarks':d.benchmark_id.nunique(),'n_components':len(components),'baseline':float(pairs.baseline_S3.mean()),'source_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),'results':{}}
 assert abs(out['baseline']-.2059838433113593)<1e-12,'control reproduction failed'
 for name in list(vals)[1:]:
  delta=(pairs[name]-pairs.baseline_S3).to_numpy()
  out['results'][name]={'brier_alc':float(pairs[name].mean()),'delta_brier_vs_S3':float(delta.mean()),'paired_component_ci95':np.quantile([delta[s].mean() for s in samples],[.025,.975]).tolist(),'improved_pairs':int((delta<0).sum()),'worsened_pairs':int((delta>0).sum())}
 out['generated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 (ROOT/'results/prior_strength_experiments.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
 pairs.to_csv(ROOT/'results/prior_strength_pair_scores.csv',index=False)
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
