import sys, time, os
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi, hf_hub_download
R="aims-foundations/measurement-db"; LOCAL=r"E:\AIMS-sprint\data"
api=HfApi()
tables=sys.argv[1].split(",")          # p.ej. responses,subjects,benchmarks
roots=[e.path for e in api.list_repo_tree(R,repo_type="dataset",recursive=False) if e.__class__.__name__=="RepoFolder"]
targets=[f"{b}/formatted_tables/{t}.parquet" for b in roots for t in tables]
def get(p):
    try:
        hf_hub_download(R,p,repo_type="dataset",local_dir=LOCAL); return ("ok",p)
    except Exception as e:
        return (type(e).__name__,p)
t0=time.time(); ok=err=0
with ThreadPoolExecutor(12) as ex:
    futs=[ex.submit(get,p) for p in targets]
    for i,f in enumerate(as_completed(futs),1):
        s,p=f.result()
        if s=="ok": ok+=1
        else:
            err+=1
            if err<=10: print("FALLO",s,p,flush=True)
        if i%100==0: print(f"{i}/{len(targets)} ok={ok} err={err} {time.time()-t0:.0f}s",flush=True)
print(f"FIN {tables}: ok={ok} err={err} en {time.time()-t0:.0f}s",flush=True)
