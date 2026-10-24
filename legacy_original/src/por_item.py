"""¿Se puede predecir POR ÍTEM en vez de dar un valor constante a todo el par?

Motivación: el oráculo constante por par obtiene 0,179 y el oráculo por ítem
0,006. El 96,7 % del suelo es variación ENTRE ÍTEMS dentro del par. Nuestro
predictor, como el baseline oficial, es constante por par, así que ese 96,7 %
está íntegro sobre la mesa. `predict(input, labeled)` recibe el ítem concreto:
nada obliga a ignorarlo.

El experimento anterior (texto_b0.py) preguntaba si los rasgos del texto predicen
el acierto ABSOLUTO, lo que mezcla la dificultad del par con la del ítem. Aquí se
pregunta lo correcto: ¿predicen la DESVIACIÓN de un ítem respecto a la media de
su par? Eso es dificultad relativa dentro de un benchmark, que es justo lo que
podría transferir a un benchmark nuevo aunque su nivel absoluto no lo haga.

Modelo: p_ij = p_par_contraido_i + lambda * f(rasgos del item j), con f ajustada
FUERA del benchmark objetivo sobre la desviación en PROBABILIDAD respecto a la
media del par (Ridge sobre el residuo, no logística con offset).

Protocolo: todo se ajusta y se elige con los 50 benchmarks de desarrollo.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import ejecuta as E
FUERZA = json.load(open(r"E:\AIMS-sprint\work\hiper.json"))["fuerza"]
from harness import BUDGETS, WEIGHTS
from predictores import BetaBinomial
from texto_b0 import rasgos_texto

DATA = r"E:\AIMS-sprint\data"
EPS = 1e-6


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sigmoide(z):
    return 1.0 / (1.0 + np.exp(-z))


def contenido_de(benchmark, item_ids):
    """Texto de unos ítems concretos de un benchmark."""
    f = os.path.join(DATA, benchmark, "formatted_tables", "items.parquet")
    if not os.path.exists(f):
        return None
    d = pd.read_parquet(f, columns=["item_id", "content"])
    d = d[d["item_id"].isin(set(item_ids))].drop_duplicates("item_id").set_index("item_id")
    return d["content"].reindex(item_ids)


def tabla_entrenamiento(benches, max_por_bench=60_000):
    """Desviación de cada respuesta respecto a la media de su par, con rasgos."""
    trozos = []
    for b in sorted(benches):
        fr = os.path.join(DATA, b, "formatted_tables", "responses.parquet")
        fi = os.path.join(DATA, b, "formatted_tables", "items.parquet")
        if not (os.path.exists(fr) and os.path.exists(fi)):
            continue
        r = pd.read_parquet(fr, columns=["subject_id", "item_id", "benchmark_id", "response"]).dropna()
        if r.empty:
            continue
        # solo pares con suficientes items, que son los que el concurso evalua
        tam = r.groupby("subject_id")["item_id"].nunique()
        r = r[r["subject_id"].isin(tam[tam >= 80].index)]
        if r.empty:
            continue
        if len(r) > max_por_bench:
            r = r.sample(max_por_bench, random_state=0)
        r["media_par"] = r.groupby("subject_id")["response"].transform("mean")
        it = pd.read_parquet(fi, columns=["item_id", "content"])
        d = r.merge(it, on="item_id", how="left")
        X = rasgos_texto(d["content"])
        X["benchmark_id"] = d["benchmark_id"].values
        X["y"] = d["response"].astype(int).values
        X["media_par"] = d["media_par"].values
        trozos.append(X)
    return pd.concat(trozos, ignore_index=True) if trozos else pd.DataFrame()


def ajusta_desviacion(D, cols):
    """Ridge sobre el RESIDUO respecto a la media del par.

    No es una logística con offset: como sigmoid(logit(p)) = p, el "offset" solo
    sirve para centrar el objetivo en `y - media_par`. Lo que se ajusta es una
    regresión lineal regularizada de ese residuo sobre los rasgos del ítem, y su
    predicción se suma en PROBABILIDAD, no en logit. La corrección de este
    docstring viene de la verificación independiente, que lo describía mal.

    `media_par` se calcula solo con benchmarks de ENTRENAMIENTO: no entra ninguna
    etiqueta del par evaluado.
    """
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(D[cols])
    Z = sc.transform(D[cols])
    off = logit(D["media_par"].to_numpy())
    # sklearn no admite offset: se ajusta sobre el residuo en dos pasos
    # (1) probabilidad base desde el offset, (2) regresion sobre el residuo
    base = sigmoide(off)
    resid = D["y"].to_numpy() - base
    from sklearn.linear_model import Ridge
    m = Ridge(alpha=50.0).fit(Z, resid)
    return sc, m


def main():
    pares, agg, dev, cab = E.carga()
    print(f"construyendo tabla de desviacion con {len(dev)} benchmarks de desarrollo...", flush=True)
    D = tabla_entrenamiento(dev)
    cols = [c for c in D.columns if c not in ("benchmark_id", "y", "media_par")]
    print(f"filas={len(D):,}  benchmarks={D['benchmark_id'].nunique()}\n")

    dev_p = [p for p in pares if p.benchmark_id in dev]
    bb = BetaBinomial(0.5, FUERZA)   # misma fuerza que el metodo entregado

    resultados = {}
    for escala in (0.0, 0.5, 1.0, 1.5):
        alc_tot = []
        for b in sorted(dev):
            ps = [p for p in dev_p if p.benchmark_id == b]
            if not ps:
                continue
            tr = D[D["benchmark_id"] != b]
            if len(tr) < 20_000:
                continue
            sc, m = ajusta_desviacion(tr, cols)
            for par in ps:
                cont = contenido_de(par.benchmark_id, par.eva_items)
                if cont is None or cont.isna().all():
                    continue
                X = sc.transform(rasgos_texto(cont.fillna(""))[cols])
                delta = m.predict(X) * escala          # desviacion en PROBABILIDAD
                por_b = {}
                for n in BUDGETS:
                    k = min(n, len(par.adq_items))
                    p0 = float(bb(par, np.arange(k))[0])
                    p = np.clip(p0 + delta, EPS, 1 - EPS)
                    por_b[n] = float(np.mean((p[par.eva_idx] - par.eva_y) ** 2))
                alc_tot.append(sum(w * por_b[n] for n, w in zip(BUDGETS, WEIGHTS)))
        resultados[escala] = float(np.mean(alc_tot))
        etiq = "(= beta-binomial puro)" if escala == 0.0 else ""
        print(f"  escala de la desviacion = {escala:<4} ALC(dev) = {resultados[escala]:.5f} {etiq}", flush=True)

    base = resultados[0.0]
    mejor = min(resultados, key=resultados.get)
    print(f"\n  base (constante por par) = {base:.5f}")
    print(f"  mejor escala = {mejor} -> {resultados[mejor]:.5f} "
          f"({resultados[mejor]-base:+.5f})")
    veredicto = ("MEJORA" if resultados[mejor] < base - 1e-4 else
                 "NO MEJORA: la desviacion por item no transfiere a un benchmark nuevo")
    print(f"  VEREDICTO: {veredicto}")
    json.dump({"alc_por_escala": {str(k): v for k, v in resultados.items()},
               "base_constante_por_par": base, "mejor_escala": mejor,
               "veredicto": veredicto, "n_filas_entrenamiento": int(len(D))},
              open("por_item.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
