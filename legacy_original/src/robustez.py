"""Robustez e incertidumbre del resultado de cabecera.

Responde a tres objeciones concretas de la ronda 2:

1. La tabla promedia por PAR y la prueba de signos por BENCHMARK. Son dos
   unidades distintas y hay que dar las dos, sin mezclarlas.
2. Los 496 pares no son 496 unidades independientes para inferir sobre
   benchmarks nuevos: los pares se agrupan dentro de su benchmark. El intervalo
   se obtiene remuestreando BENCHMARKS (bootstrap por conglomerados).
3. Una sola muestra de adquisición puede fabricar una mejora. Se repite la
   evaluación con varias semillas de adquisición distintas.
"""
from __future__ import annotations

# pickle: cache local propia generada por construye.py en esta misma maquina
import json
import pickle

import numpy as np
import pandas as pd

import ejecuta as E
from harness import BUDGETS, WEIGHTS, brier_alc_par, semilla_estable
from predictores import (BetaBinomial, EmpiricoBayes, EmpiricoBayesSujeto,
                         constante_mitad, media_empirica)

W = r"E:\AIMS-sprint\work"
FUERZA = json.load(open(rf"{W}\hiper.json"))["fuerza"]
N_BOOT = 2000
SEMILLAS_ADQ = (1, 2, 3, 4, 5)


def evalua_con_orden(pares, predictor, semilla):
    """Igual que harness.evalua pero con un orden de adquisición remuestreado.

    La partición adquisición/evaluación NO cambia (es fija por las bases); lo que
    cambia es qué etiquetas revela la política aleatoria por defecto.
    """
    filas = []
    for par in pares:
        r = np.random.default_rng(semilla_estable(semilla, par.subject_id, par.benchmark_id))
        orden = r.permutation(len(par.adq_items))
        pb = {}
        for n in BUDGETS:
            pb[n] = predictor(par, orden[:min(n, len(orden))])
        alc, por_b = brier_alc_par(pb, par)
        filas.append({"subject_id": par.subject_id, "benchmark_id": par.benchmark_id,
                      "brier_alc": alc, **{f"B{n}": por_b[n] for n in BUDGETS}})
    return pd.DataFrame(filas)


def boot_benchmarks(res_a, res_b, n_boot=N_BOOT, semilla=7):
    """Bootstrap por conglomerados: se remuestrean BENCHMARKS, no pares.

    Devuelve el intervalo de la diferencia de la media POR PAR (a - b), que es
    la unidad en la que puntúa el concurso, pero propagando la dependencia de
    los pares dentro de su benchmark.
    """
    a = res_a.set_index(["benchmark_id", "subject_id"])["brier_alc"]
    b = res_b.set_index(["benchmark_id", "subject_id"])["brier_alc"]
    comun = a.index.intersection(b.index)
    a, b = a[comun], b[comun]
    bench = np.array([i[0] for i in comun])
    unicos = np.unique(bench)
    por_bench = {u: np.flatnonzero(bench == u) for u in unicos}
    av, bv = a.to_numpy(), b.to_numpy()
    rng = np.random.default_rng(semilla)
    difs = np.empty(n_boot)
    for i in range(n_boot):
        sel = rng.choice(unicos, len(unicos), replace=True)
        idx = np.concatenate([por_bench[u] for u in sel])
        difs[i] = av[idx].mean() - bv[idx].mean()
    return {"dif_media_por_par": float(av.mean() - bv.mean()),
            "ic95_bajo": float(np.percentile(difs, 2.5)),
            "ic95_alto": float(np.percentile(difs, 97.5)),
            "p_boot_dos_colas": float(2 * min((difs >= 0).mean(), (difs <= 0).mean())),
            "n_benchmarks": int(len(unicos)), "n_pares": int(len(av))}


def main():
    pares, agg, dev, cab = E.carga()
    cab_p = [p for p in pares if p.benchmark_id in cab]
    print(f"cabecera: {len(cab_p)} pares en {len(cab)} benchmarks\n")

    metodos = {
        "constante_0.5": lambda: constante_mitad,
        "media_empirica_OFICIAL": lambda: media_empirica,
        "beta_binomial_ENTREGADO": lambda: BetaBinomial(0.5, FUERZA),
        "empirico_bayes": lambda: EmpiricoBayes(),
        "empirico_bayes_sujeto": lambda: EmpiricoBayesSujeto(),
    }

    print("1) Estabilidad frente a la muestra de adquisicion (5 semillas)")
    por_semilla = {k: [] for k in metodos}
    guarda = {}
    for s in SEMILLAS_ADQ:
        for nombre, c in metodos.items():
            # ajuste LOBO por benchmark, igual que en ejecuta.lobo
            trozos = []
            for b in sorted(cab):
                ps = [p for p in cab_p if p.benchmark_id == b]
                if not ps:
                    continue
                pred = c()
                if hasattr(pred, "ajusta"):
                    pred.ajusta(agg[agg["benchmark_id"] != b])
                trozos.append(evalua_con_orden(ps, pred, s))
            r = pd.concat(trozos, ignore_index=True)
            por_semilla[nombre].append(float(r["brier_alc"].mean()))
            if s == SEMILLAS_ADQ[0]:
                guarda[nombre] = r
        print(f"   semilla {s}: " + "  ".join(
            f"{k.split('_')[0]}={por_semilla[k][-1]:.5f}" for k in metodos), flush=True)
    print()
    tabla_s = {k: {"media": float(np.mean(v)), "min": float(np.min(v)),
                   "max": float(np.max(v)), "sd": float(np.std(v))}
               for k, v in por_semilla.items()}
    for k, v in tabla_s.items():
        print(f"   {k:<26} ALC medio={v['media']:.5f}  rango=[{v['min']:.5f},{v['max']:.5f}]  sd={v['sd']:.5f}")

    print("\n2) Las dos unidades de promediado, sin mezclarlas")
    filas = []
    for k, r in guarda.items():
        por_par = float(r["brier_alc"].mean())
        por_bench = float(r.groupby("benchmark_id")["brier_alc"].mean().mean())
        filas.append({"predictor": k, "media_por_par": por_par, "media_por_benchmark": por_bench,
                      **{f"B{n}": float(r[f"B{n}"].mean()) for n in BUDGETS}})
    t = pd.DataFrame(filas)
    print(t.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print("\n3) Intervalos por bootstrap de BENCHMARKS (2000 remuestreos)")
    comp = {}
    base = guarda["media_empirica_OFICIAL"]
    for k in ("beta_binomial_ENTREGADO", "empirico_bayes", "constante_0.5"):
        comp[f"{k}_vs_oficial"] = boot_benchmarks(guarda[k], base)
        c = comp[f"{k}_vs_oficial"]
        print(f"   {k:<26} vs oficial: dif={c['dif_media_por_par']:+.5f} "
              f"IC95=[{c['ic95_bajo']:+.5f},{c['ic95_alto']:+.5f}] p={c['p_boot_dos_colas']:.4f}")
    for k in ("empirico_bayes", "empirico_bayes_sujeto"):
        c = boot_benchmarks(guarda[k], guarda["beta_binomial_ENTREGADO"])
        comp[f"{k}_vs_entregado"] = c
        print(f"   {k:<26} vs entregado: dif={c['dif_media_por_par']:+.5f} "
              f"IC95=[{c['ic95_bajo']:+.5f},{c['ic95_alto']:+.5f}] p={c['p_boot_dos_colas']:.4f}")

    print("\n4) Recorrido alcanzable: suelo de un predictor constante por par")
    # El suelo se calcula sobre LOS MISMOS pares de cabecera que la tabla, no
    # sobre los 769 del conjunto completo. Lo señaló la verificación independiente:
    # mezclar los dos conjuntos inflaba el porcentaje de margen capturado.
    piso = float(np.mean([np.mean((p.eva_y.mean() - p.eva_y) ** 2)
                          for p in cab_p if len(p.eva_y)]))
    of = float(guarda["media_empirica_OFICIAL"]["brier_alc"].mean())
    nu = float(guarda["beta_binomial_ENTREGADO"]["brier_alc"].mean())
    frac = (of - nu) / (of - piso)
    print(f"   suelo (oraculo por par) = {piso:.5f}")
    print(f"   oficial = {of:.5f}   entregado = {nu:.5f}")
    print(f"   recorrido disponible = {of - piso:.5f}; capturado = {of - nu:.5f} "
          f"-> {100*frac:.1f} % del margen alcanzable por un predictor constante por par")

    json.dump({"estabilidad_semillas": tabla_s,
               "dos_unidades": t.to_dict("records"),
               "bootstrap": comp,
               "recorrido": {"suelo_oraculo_par": piso, "oficial": of, "entregado": nu,
                             "fraccion_capturada": frac}},
              open(rf"{W}\robustez.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\nguardado robustez.json")


if __name__ == "__main__":
    main()
