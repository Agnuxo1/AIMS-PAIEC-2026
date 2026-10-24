"""Evaluación LOBO (leave-one-benchmark-out) de todos los predictores.

Separa los benchmarks en DESARROLLO (se ajustan ahí los hiperparámetros) y
CABECERA (nunca se mira hasta el final). Así la cifra que va al informe no se
elige sobre los mismos pliegues que la produjeron.

    python ejecuta.py tune     -> curva de `fuerza` solo en desarrollo
    python ejecuta.py final    -> tabla final en cabecera, con la fuerza elegida
"""
from __future__ import annotations

import json
# pickle: solo cache LOCAL de este sprint, generada por construye.py en esta
# misma maquina a partir del dataset oficial. No entra en el ZIP de entrega ni
# se carga nada de origen externo. El paquete de entrega no usa pickle.
import pickle
import sys

import numpy as np
import pandas as pd

from harness import BUDGETS, evalua, puntuacion
from predictores import (BetaBinomial, EmpiricoBayes, EmpiricoBayesSujeto, Jerarquico,
                         JerarquicoMacro, SoloGlobal, SoloIdentidad, constante_mitad,
                         media_empirica)

SEMILLA = 20261024
W = r"E:\AIMS-sprint\work"


def carga():
    with open(rf"{W}\pares.pkl", "rb") as f:
        pares = pickle.load(f)
    agg = pd.read_parquet(rf"{W}\agregados.parquet")
    benches = sorted({p.benchmark_id for p in pares})
    rng = np.random.default_rng(SEMILLA)
    perm = rng.permutation(len(benches))
    n_dev = 50
    dev = {benches[i] for i in perm[:n_dev]}
    cab = {benches[i] for i in perm[n_dev:]}
    return pares, agg, dev, cab


def lobo(pares, agg, benches, construir):
    """Evalúa pares de cada benchmark con el modelo ajustado SIN ese benchmark."""
    filas = []
    for b in sorted(benches):
        ps = [p for p in pares if p.benchmark_id == b]
        if not ps:
            continue
        pred = construir()
        if hasattr(pred, "ajusta"):
            pred.ajusta(agg[agg["benchmark_id"] != b])
        r = evalua(ps, pred)
        filas.append(r)
    return pd.concat(filas, ignore_index=True) if filas else pd.DataFrame()


def resumen(nombre, res):
    fila = {"predictor": nombre, "brier_alc": puntuacion(res),
            "n_pares": len(res), "n_benchmarks": res["benchmark_id"].nunique()}
    for n in BUDGETS:
        fila[f"B{n}"] = float(res[f"B{n}"].mean())
    return fila


def por_benchmark(res):
    return res.groupby("benchmark_id")["brier_alc"].mean()


def prueba_signos(a: pd.Series, b: pd.Series):
    """Diferencias emparejadas por benchmark + prueba de intercambio de signos.

    Con pocos benchmarks la resolución es escasa; se reporta junto al tamaño del
    efecto, que es lo que manda (ver PLAN.md 2.D).
    """
    comun = a.index.intersection(b.index)
    d = (a[comun] - b[comun]).to_numpy()
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return {"n": 0}
    gana = int((d < 0).sum())  # menor Brier = mejor para `a`
    from math import comb
    k = min(gana, n - gana)
    p = sum(comb(n, i) for i in range(k + 1)) * 2 / 2 ** n
    return {"n": n, "gana_a": gana, "p_bilateral": min(1.0, p),
            "media_dif": float(d.mean()), "mediana_dif": float(np.median(d))}


def main():
    modo = sys.argv[1] if len(sys.argv) > 1 else "final"
    pares, agg, dev, cab = carga()
    print(f"pares={len(pares)} benchmarks: desarrollo={len(dev)} cabecera={len(cab)}\n")

    if modo == "tune":
        print("Curva de la fuerza de contraccion, SOLO en desarrollo")
        print(f"{'fuerza':>7} {'ALC beta-binom':>15} {'ALC jerarq(resp)':>15} {'ALC jerarq(macro)':>17}")
        out = []
        for f in (1, 2, 4, 6, 8, 12, 16, 24, 32, 48):
            bb = puntuacion(lobo(pares, agg, dev, lambda f=f: BetaBinomial(0.5, f)))
            je = puntuacion(lobo(pares, agg, dev, lambda f=f: Jerarquico(fuerza=f)))
            jm = puntuacion(lobo(pares, agg, dev, lambda f=f: JerarquicoMacro(fuerza=f)))
            print(f"{f:>7} {bb:>15.5f} {je:>15.5f} {jm:>15.5f}", flush=True)
            out.append({"fuerza": f, "beta_binomial": bb, "jerarquico": je, "jerarquico_macro": jm})
        pd.DataFrame(out).to_csv(rf"{W}\curva_fuerza.csv", index=False)
        mejor = min(out, key=lambda r: r["jerarquico"])
        print(f"\nmejor fuerza en desarrollo: {mejor['fuerza']} (ALC {mejor['jerarquico']:.5f})")
        # NO escribe hiper.json: la fuerza entregada la fija barrido_prior.py con
        # el argmin entre las variantes beta-binomiales. Este barrido explora las
        # variantes JERARQUICAS, cuyo argmin es distinto; si escribiera aqui, el
        # pipeline publicado no reproduciria la fuerza entregada.
        return

    F = json.load(open(rf"{W}\hiper.json"))["fuerza"]
    print(f"fuerza elegida en desarrollo: {F}\n")
    metodos = {
        "constante_0.5": lambda: constante_mitad,
        "media_empirica_OFICIAL": lambda: media_empirica,
        "beta_binomial_p0=0.5": lambda: BetaBinomial(0.5, F),
        "jerarquico_resp": lambda: Jerarquico(fuerza=F),
        "jerarquico_macro": lambda: JerarquicoMacro(fuerza=F),
        "empirico_bayes": lambda: EmpiricoBayes(),
        "empirico_bayes_sujeto": lambda: EmpiricoBayesSujeto(),
        "ablacion_solo_identidad": lambda: SoloIdentidad(fuerza=F),
        "ablacion_solo_global": lambda: SoloGlobal(fuerza=F),
    }
    res, tabla = {}, []
    for nombre, c in metodos.items():
        r = lobo(pares, agg, cab, c)
        res[nombre] = r
        tabla.append(resumen(nombre, r))
        print(f"  {nombre:<26} ALC={tabla[-1]['brier_alc']:.5f}", flush=True)

    t = pd.DataFrame(tabla).sort_values("brier_alc")
    t.to_csv(rf"{W}\resultados_cabecera.csv", index=False)
    print("\n" + t.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print("\nComparaciones emparejadas por benchmark (menor Brier ALC = mejor):")
    base = por_benchmark(res["media_empirica_OFICIAL"])
    comps = {}
    for nombre in ("empirico_bayes", "empirico_bayes_sujeto", "beta_binomial_p0=0.5", "jerarquico_macro", "constante_0.5"):
        s = prueba_signos(por_benchmark(res[nombre]), base)
        comps[f"{nombre}_vs_oficial"] = s
        print(f"  {nombre:<24} vs oficial: gana en {s['gana_a']}/{s['n']} benchmarks, "
              f"dif media={s['media_dif']:+.5f}, p={s['p_bilateral']:.2e}")
    s = prueba_signos(por_benchmark(res["empirico_bayes"]), por_benchmark(res["beta_binomial_p0=0.5"]))
    comps["empirico_bayes_vs_beta_binomial"] = s
    print(f"  {'empirico_bayes':<24} vs beta-binomial (control fuerte): gana en "
          f"{s['gana_a']}/{s['n']}, dif media={s['media_dif']:+.5f}, p={s['p_bilateral']:.2e}")
    json.dump(comps, open(rf"{W}\comparaciones.json", "w"), indent=1)

    for nombre, r in res.items():
        r.to_parquet(rf"{W}\res_{nombre.replace('.','').replace('=','')}.parquet")
    print("\nguardado: resultados_cabecera.csv, comparaciones.json, res_*.parquet")


if __name__ == "__main__":
    main()
