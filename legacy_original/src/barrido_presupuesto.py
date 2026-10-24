"""Barrido de la fuerza de contraccion presupuesto a presupuesto (SOLO desarrollo).

Faltaba el generador de fuerza_por_presupuesto.json; lo senalo la verificacion
independiente. Como el predictor es constante por par, el Brier de cada
presupuesto es independiente y puede optimizarse por separado.
"""
import json
import numpy as np
import ejecuta as E
from harness import BUDGETS, WEIGHTS
from predictores import BetaBinomial

REJILLA = [0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24]

def main():
    pares, agg, dev, cab = E.carga()
    dev_p = [p for p in pares if p.benchmark_id in dev]
    print(f"pares de desarrollo: {len(dev_p)}")
    mejor, tabla = {}, {}
    print(f"\n{'n':>3} " + " ".join(f"{f:>7}" for f in REJILLA))
    for n in BUDGETS:
        fila = []
        for f in REJILLA:
            bb = BetaBinomial(0.5, f)
            fila.append(float(np.mean([
                np.mean((bb(par, np.arange(min(n, len(par.adq_items))))[par.eva_idx] - par.eva_y) ** 2)
                for par in dev_p])))
        i = int(np.argmin(fila)); mejor[n] = REJILLA[i]; tabla[n] = fila
        nota = "  (degenerado: sin etiquetas, toda S da lo mismo)" if n == 0 else ""
        print(f"{n:>3} " + " ".join(f"{v:7.4f}" for v in fila) + f"  -> S={REJILLA[i]}{nota}")
    alc_var = sum(w * min(tabla[n]) for n, w in zip(BUDGETS, WEIGHTS))
    def alc_fijo(f):
        return sum(w * tabla[n][REJILLA.index(f)] for n, w in zip(BUDGETS, WEIGHTS))
    print(f"\nALC(dev) con S por presupuesto = {alc_var:.5f}")
    for f in (3, 4, 5):
        print(f"ALC(dev) con S={f} fijo        = {alc_fijo(f):.5f}")
    json.dump({"por_presupuesto": {str(k): v for k, v in mejor.items()},
               "alc_variable": alc_var, "alc_fijo": {str(f): alc_fijo(f) for f in (3, 4, 5)},
               "rejilla": REJILLA, "tabla": {str(k): v for k, v in tabla.items()}},
              open("fuerza_por_presupuesto.json", "w"), indent=1)
    print("guardado fuerza_por_presupuesto.json")

if __name__ == "__main__":
    main()
