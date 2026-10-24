"""Construye el conjunto de validación local: pares sujeto-benchmark + agregados.

Procesa un benchmark cada vez para no cargar 70 M de respuestas en memoria.
Guarda:
  - pares.pkl      : lista de Par muestreados (el conjunto de evaluación local)
  - agregados.parquet : (subject_id, benchmark_id, k, n) de TODAS las respuestas
                        binarias públicas, que es lo que ajusta los priors.
"""
from __future__ import annotations

# pickle: solo cache LOCAL de este sprint, generada por construye.py en esta
# misma maquina a partir del dataset oficial. No entra en el ZIP de entrega ni
# se carga nada de origen externo. El paquete de entrega no usa pickle.
import pickle
import sys
import time

import numpy as np
import pandas as pd

from harness import MIN_ITEMS, benchmarks_binarios, carga_respuestas, construye_pares

PARES_POR_BENCH = 6      # pares muestreados por benchmark
MAX_EVA_ITEMS = 150      # tope de ítems distintos de evaluación por par (coste)
SEMILLA = 20261024


def recorta(par, rng):
    """Limita los ítems de evaluación de un par para acotar coste y memoria.

    Es una aproximación LOCAL declarada en el informe: la evaluación oficial no
    recorta. No afecta a la adquisición ni al anidamiento, solo al tamaño del
    pool de evaluación sobre el que se promedia el Brier del par.
    """
    if len(par.eva_items) <= MAX_EVA_ITEMS:
        return par
    sel = np.sort(rng.choice(len(par.eva_items), MAX_EVA_ITEMS, replace=False))
    mapa = {viejo: nuevo for nuevo, viejo in enumerate(sel)}
    keep = np.isin(par.eva_idx, sel)
    par.eva_items = par.eva_items[sel]
    par.eva_y = par.eva_y[keep]
    par.eva_idx = np.array([mapa[i] for i in par.eva_idx[keep]], dtype=np.int32)
    return par


def main():
    cat = benchmarks_binarios()
    nombres = sorted(cat["benchmark_id"].tolist())
    print(f"benchmarks binarios publicos en catalogo: {len(nombres)}")

    pares, agg, saltados = [], [], []
    rng = np.random.default_rng(SEMILLA)
    t0 = time.time()
    for i, b in enumerate(nombres, 1):
        d = carga_respuestas(b)
        if d is None or d.empty:
            saltados.append((b, "sin datos o no binario")); continue
        g = d.groupby(["subject_id", "benchmark_id"], sort=False)["y"].agg(["sum", "count"])
        g.columns = ["k", "n"]
        agg.append(g.reset_index())

        ps = construye_pares(d, semilla=SEMILLA, max_pares=PARES_POR_BENCH)
        if not ps:
            saltados.append((b, f"ningun par con >={MIN_ITEMS} items")); continue
        pares += [recorta(p, rng) for p in ps]
        del d
        if i % 25 == 0:
            print(f"  {i}/{len(nombres)} pares={len(pares)} {time.time()-t0:.0f}s", flush=True)

    agg = pd.concat(agg, ignore_index=True)
    agg.to_parquet(r"E:\AIMS-sprint\work\agregados.parquet")
    with open(r"E:\AIMS-sprint\work\pares.pkl", "wb") as f:
        pickle.dump(pares, f)

    nb = len({p.benchmark_id for p in pares})
    print(f"\nPARES: {len(pares)} en {nb} benchmarks")
    print(f"AGREGADOS: {len(agg)} filas sujeto-benchmark, {int(agg['n'].sum()):,} respuestas")
    print(f"respuestas de evaluacion en los pares: {sum(len(p.eva_y) for p in pares):,}")
    print(f"saltados: {len(saltados)}")
    for b, r in saltados[:12]:
        print("   ", b, "->", r)


if __name__ == "__main__":
    sys.exit(main())
