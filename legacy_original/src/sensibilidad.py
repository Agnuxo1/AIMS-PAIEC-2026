"""Sensibilidad del resultado principal a las dos aproximaciones locales:
el tope de items de evaluacion por par y cuantos pares se muestrean por benchmark.
Tambien descompone el suelo del oraculo por par en variacion ENTRE ITEMS y
variacion ENTRE REPETICIONES del mismo item, que el informe habia confundido.
"""
# pickle: cache local propia generada por construye.py en esta misma maquina
import pickle, json, copy
import numpy as np, pandas as pd
import ejecuta as E
FUERZA = json.load(open(r"E:\AIMS-sprint\work\hiper.json"))["fuerza"]
from harness import BUDGETS, WEIGHTS, brier_alc_par
from predictores import BetaBinomial, media_empirica

def recorta(par, tope, rng):
    p = copy.copy(par)
    if len(p.eva_items) <= tope: return p
    sel = np.sort(rng.choice(len(p.eva_items), tope, replace=False))
    mapa = {v:i for i,v in enumerate(sel)}
    keep = np.isin(p.eva_idx, sel)
    p.eva_items = p.eva_items[sel]; p.eva_y = p.eva_y[keep]
    p.eva_idx = np.array([mapa[i] for i in p.eva_idx[keep]], dtype=np.int32)
    return p

def corre(pares, agg, benches, pred_f):
    tot=[]
    for b in sorted(benches):
        ps=[p for p in pares if p.benchmark_id==b]
        if not ps: continue
        pr=pred_f()
        if hasattr(pr,"ajusta"): pr.ajusta(agg[agg["benchmark_id"]!=b])
        for par in ps:
            pb={n:pr(par,np.arange(min(n,len(par.adq_items)))) for n in BUDGETS}
            tot.append(brier_alc_par(pb,par)[0])
    return float(np.mean(tot))

pares,agg,dev,cab=E.carga()
cab_p=[p for p in pares if p.benchmark_id in cab]
out={}
print("1) Sensibilidad al tope de items de evaluacion por par")
rng=np.random.default_rng(11)
for tope in (25,50,100,150):
    sub=[recorta(p,tope,rng) for p in cab_p]
    of=corre(sub,agg,cab,lambda: media_empirica); nu=corre(sub,agg,cab,lambda: BetaBinomial(0.5, FUERZA))
    out[f"tope_{tope}"]={"oficial":of,"entregado":nu,"dif":nu-of,"rel":(of-nu)/of}
    print(f"   tope={tope:>3}: oficial={of:.5f} entregado={nu:.5f} dif={nu-of:+.5f} ({100*(of-nu)/of:.1f}%)",flush=True)

print("\n2) Sensibilidad al numero de pares por benchmark (submuestreo)")
r2=np.random.default_rng(12)
for k in (1,2,4,6):
    sel=[]
    for b in sorted(cab):
        ps=[p for p in cab_p if p.benchmark_id==b]
        if not ps: continue
        idx=r2.choice(len(ps),min(k,len(ps)),replace=False)
        sel+= [ps[i] for i in idx]
    of=corre(sel,agg,cab,lambda: media_empirica); nu=corre(sel,agg,cab,lambda: BetaBinomial(0.5, FUERZA))
    out[f"pares_{k}"]={"n_pares":len(sel),"oficial":of,"entregado":nu,"dif":nu-of,"rel":(of-nu)/of}
    print(f"   {k} par(es)/bench ({len(sel):>3} pares): oficial={of:.5f} entregado={nu:.5f} ({100*(of-nu)/of:.1f}%)",flush=True)

print("\n3) Descomposicion del suelo del oraculo por par")
ent=[];item=[];par_=[]
for p in cab_p:
    y=p.eva_y
    if len(y)==0: continue
    m_item=pd.Series(y).groupby(p.eva_idx).transform("mean").to_numpy()
    m_par=y.mean()
    par_.append(float(np.mean((m_par-y)**2)))          # oraculo por par
    item.append(float(np.mean((m_item-y)**2)))         # oraculo por item (ruido de repeticion)
    ent.append(float(np.mean((m_par-m_item)**2)))      # varianza ENTRE items dentro del par
op,oi,ei=map(np.mean,(par_,item,ent))
out["descomposicion"]={"oraculo_por_par":op,"oraculo_por_item_ruido_repeticion":oi,
                       "varianza_entre_items":ei,"suma":oi+ei,
                       "pct_entre_items":100*ei/op,"pct_repeticion":100*oi/op}
print(f"   oraculo por par            = {op:.5f}")
print(f"   ruido de repeticion (item) = {oi:.5f}  ({100*oi/op:.1f} % del suelo)")
print(f"   varianza entre items       = {ei:.5f}  ({100*ei/op:.1f} % del suelo)")
print(f"   comprobacion suma          = {oi+ei:.5f}")

print("\n4) Distribucion de la diferencia por benchmark (entregado - oficial)")
difs={}
for b in sorted(cab):
    ps=[p for p in cab_p if p.benchmark_id==b]
    if not ps: continue
    of=corre(ps,agg,[b],lambda: media_empirica); nu=corre(ps,agg,[b],lambda: BetaBinomial(0.5, FUERZA))
    difs[b]=nu-of
d=pd.Series(difs)
out["por_benchmark"]={"n":int(len(d)),"mejora_en":int((d<0).sum()),"empata":int((d==0).sum()),
  "empeora_en":int((d>0).sum()),"media":float(d.mean()),"mediana":float(d.median()),
  "p10":float(d.quantile(.10)),"p90":float(d.quantile(.90)),"peor":float(d.max()),"mejor":float(d.min()),
  "top5_mejores":d.nsmallest(5).to_dict(),"top5_peores":d.nlargest(5).to_dict()}
print(f"   mejora en {int((d<0).sum())}/{len(d)}, empeora en {int((d>0).sum())}")
print(f"   mediana={d.median():+.5f} p10={d.quantile(.10):+.5f} p90={d.quantile(.90):+.5f}")
print(f"   mejor={d.min():+.5f} peor={d.max():+.5f}")
sin_top5 = d.drop(d.nsmallest(5).index)
out["sin_los_5_mejores"]=float(sin_top5.mean())
print(f"   media sin los 5 benchmarks donde mas gana: {sin_top5.mean():+.5f} (total {d.mean():+.5f})")
print("\n5) Reponderacion por pares elegibles (el oficial pondera por par, no por benchmark)")
# Muestrear 6 pares por benchmark da el mismo peso a cada benchmark; el concurso
# promedia pares. Si unos pocos benchmarks concentran la mayoria de pares
# elegibles, nuestro proxy los infrapondera. Lo senalo la verificacion independiente.
eleg = agg[agg["n"] >= 80].groupby("benchmark_id").size()
eleg = eleg[eleg.index.isin(cab)]
tot_eleg = int(eleg.sum())
conc = (eleg.sort_values(ascending=False) / tot_eleg).head(5)
print(f"   pares elegibles en cabecera: {tot_eleg}; los 5 mayores concentran "
      f"{100*conc.sum():.1f} %: {dict(conc.round(3))}")
difs_s = pd.Series(difs)
w = eleg.reindex(difs_s.index).fillna(0.0)
dif_pond = float((difs_s * w).sum() / w.sum()) if w.sum() else float("nan")
of_pond = None
out["reponderado_por_pares_elegibles"] = {
    "pares_elegibles_cabecera": tot_eleg,
    "concentracion_top5": {k: float(v) for k, v in conc.items()},
    "dif_media_sin_ponderar": float(difs_s.mean()),
    "dif_media_ponderada_por_pares_elegibles": dif_pond}
print(f"   diferencia media sin ponderar = {difs_s.mean():+.5f}")
print(f"   diferencia media ponderada    = {dif_pond:+.5f}")

json.dump(out,open("sensibilidad.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
print("\nguardado sensibilidad.json")
