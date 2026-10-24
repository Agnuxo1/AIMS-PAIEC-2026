"""¿Una representación del texto rica cambia la conclusión del experimento por ítem?

La ronda 2 de crítica señaló, con razón, que doce rasgos hechos a mano acotan mal
la afirmación «el texto no ayuda». Aquí se repite el experimento de desviación por
ítem con una representación dispersa de texto con hashing (palabras + n-gramas de caracteres, 2^18 + 2^17 rasgos),
que es barata, no necesita modelos grandes y es mucho más expresiva.

Se mantiene el mismo protocolo: objetivo = desviación de la respuesta respecto a
la media de SU par, modelo ajustado FUERA del benchmark objetivo, validación
dejando fuera un benchmark entero dentro de los 50 de desarrollo.

Nota sobre la alternativa descartada: el repositorio companion de embeddings de
ítems `measurement-db-embed` solo publica 3 de los 165 benchmarks binarios
(real_webagents, researchcodebench, swe_rebench), insuficiente para medir
transferencia entre benchmarks.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import ejecuta as E

DATA = r"E:\AIMS-sprint\data"
MAX_FILAS_POR_BENCH = 4_000
MAX_CHARS = 2_000


def carga(benches):
    trozos = []
    for b in sorted(benches):
        fr = os.path.join(DATA, b, "formatted_tables", "responses.parquet")
        fi = os.path.join(DATA, b, "formatted_tables", "items.parquet")
        if not (os.path.exists(fr) and os.path.exists(fi)):
            continue
        r = pd.read_parquet(fr, columns=["subject_id", "item_id", "benchmark_id", "response"]).dropna()
        if r.empty:
            continue
        tam = r.groupby("subject_id")["item_id"].nunique()
        r = r[r["subject_id"].isin(tam[tam >= 80].index)]
        if r.empty:
            continue
        if len(r) > MAX_FILAS_POR_BENCH:
            r = r.sample(MAX_FILAS_POR_BENCH, random_state=0)
        r["media_par"] = r.groupby("subject_id")["response"].transform("mean")
        it = pd.read_parquet(fi, columns=["item_id", "content"])
        d = r.merge(it, on="item_id", how="left")
        d["content"] = d["content"].fillna("").astype(str).str.slice(0, MAX_CHARS)
        trozos.append(d[["benchmark_id", "content", "response", "media_par"]])
    return pd.concat(trozos, ignore_index=True) if trozos else pd.DataFrame()


def main():
    _, _, dev, _ = E.carga()
    print(f"cargando texto de {len(dev)} benchmarks de desarrollo...", flush=True)
    D = carga(dev)
    print(f"filas={len(D):,}  benchmarks={D['benchmark_id'].nunique()}\n", flush=True)

    from scipy import sparse
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.linear_model import SGDRegressor

    benches = sorted(D["benchmark_id"].unique())
    filas = []
    for b in benches:
        tr, te = D[D["benchmark_id"] != b], D[D["benchmark_id"] == b]
        if len(te) < 300 or len(tr) < 20_000:
            continue
        if len(tr) > 40_000:
            tr = tr.sample(40_000, random_state=1)
        # HashingVectorizer: sin vocabulario en memoria, imprescindible aqui porque
        # la maquina tiene poca RAM libre. 2^18 rasgos de palabra + 2^17 de caracter.
        vec_w = HashingVectorizer(n_features=2**18, ngram_range=(1, 2),
                                  alternate_sign=False, norm="l2",
                                  strip_accents="unicode")
        vec_c = HashingVectorizer(n_features=2**17, analyzer="char_wb",
                                  ngram_range=(3, 4), alternate_sign=False, norm="l2")
        Xtr = sparse.hstack([vec_w.transform(tr["content"]),
                             vec_c.transform(tr["content"])]).tocsr()
        Xte = sparse.hstack([vec_w.transform(te["content"]),
                             vec_c.transform(te["content"])]).tocsr()
        ytr = (tr["response"] - tr["media_par"]).to_numpy()
        m = SGDRegressor(loss="squared_error", penalty="l2", alpha=1e-5,
                         max_iter=12, tol=None, random_state=0).fit(Xtr, ytr)
        delta = m.predict(Xte)
        base = te["media_par"].to_numpy()
        y = te["response"].to_numpy()
        fila = {"benchmark": b, "n": int(len(te)),
                "brier_base": float(np.mean((base - y) ** 2))}
        for lam in (0.5, 1.0):
            p = np.clip(base + lam * delta, 1e-6, 1 - 1e-6)
            fila[f"brier_lam{lam}"] = float(np.mean((p - y) ** 2))
        # correlacion entre la desviacion predicha y la real: ¿hay señal alguna?
        real = y - base
        fila["corr"] = float(np.corrcoef(delta, real)[0, 1]) if delta.std() > 0 else 0.0
        filas.append(fila)
        print(f"  {b:<34} base={fila['brier_base']:.5f} "
              f"l0.5={fila['brier_lam0.5']:.5f} l1={fila['brier_lam1.0']:.5f} "
              f"corr={fila['corr']:+.4f}", flush=True)

    t = pd.DataFrame(filas)
    bb, b05, b10 = t["brier_base"].mean(), t["brier_lam0.5"].mean(), t["brier_lam1.0"].mean()
    gana05 = int((t["brier_lam0.5"] < t["brier_base"]).sum())
    n = len(t)
    from math import comb
    k = min(gana05, n - gana05)
    pval = min(1.0, sum(comb(n, i) for i in range(k + 1)) * 2 / 2 ** n) if n else 1.0
    print(f"\nORÁCULO DE PARTIDA (media verdadera del par): Brier = {bb:.5f}")
    print(f"  + TF-IDF, lambda=0.5: {b05:.5f}  ({b05-bb:+.5f}), gana en {gana05}/{n}, p={pval:.4f}")
    print(f"  + TF-IDF, lambda=1.0: {b10:.5f}  ({b10-bb:+.5f})")
    print(f"  correlación media entre desviación predicha y real: {t['corr'].mean():+.4f} "
          f"(mediana {t['corr'].median():+.4f}, máx {t['corr'].max():+.4f})")
    ver = ("HAY senal de item transferible" if b05 < bb and pval < 0.05
           else "NO se detecta senal de item transferible ni con TF-IDF")
    print(f"VEREDICTO: {ver}")
    json.dump({"n_benchmarks": n, "filas_entrenamiento": int(len(D)),
               "brier_base_oraculo_par": float(bb), "brier_lambda_0.5": float(b05),
               "brier_lambda_1.0": float(b10), "gana_lambda_0.5": gana05, "p": float(pval),
               "corr_media": float(t["corr"].mean()), "corr_max": float(t["corr"].max()),
               "veredicto": ver, "por_benchmark": t.to_dict("records")},
              open("tfidf_item.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
