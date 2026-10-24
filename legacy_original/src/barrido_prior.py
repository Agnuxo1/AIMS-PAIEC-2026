"""Variantes del prior empirico no parametrico (SOLO desarrollo).

Faltaba el generador de estos resultados; lo senalo la verificacion
independiente. Barre filtro por tamano minimo del par, recentrado de la media
del prior y peso de la mezcla uniforme.
"""
import json
import numpy as np
import ejecuta as E
from harness import puntuacion
from predictores import BetaBinomial, EmpiricoBayes, EmpiricoBayesSujeto

class EBvar(EmpiricoBayes):
    def __init__(self, min_n=80, mezcla=0.05, centrar=None, n_rejilla=101):
        super().__init__(n_rejilla=n_rejilla, mezcla_uniforme=mezcla, min_n=min_n)
        self.centrar = centrar
    def ajusta(self, tabla):
        super().ajusta(tabla)
        if self.centrar is not None:
            m = float((self.w * self.grid).sum())
            g = np.clip(self.grid + (self.centrar - m), 1e-3, 1 - 1e-3)
            o = np.argsort(g); self.grid, self.w = g[o], self.w[o]
        return self

def main():
    pares, agg, dev, _ = E.carga()
    pruebas = [("beta_binomial S=3", lambda: BetaBinomial(0.5, 3)),
               ("beta_binomial S=4", lambda: BetaBinomial(0.5, 4)),
               ("beta_binomial S=5", lambda: BetaBinomial(0.5, 5)),
               ("EB_sujeto", lambda: EmpiricoBayesSujeto())]
    for mn in (80, 500, 5000):
        for ce in (None, 0.5):
            pruebas.append((f"EB min_n={mn} centrar={ce}", lambda mn=mn, ce=ce: EBvar(min_n=mn, centrar=ce)))
    for mz in (0.15, 0.3):
        pruebas.append((f"EB mezcla={mz} centrar=0.5", lambda mz=mz: EBvar(mezcla=mz, centrar=0.5)))
    res = {}
    print(f"{'variante':<42}{'ALC(dev)':>10}")
    for n, c in pruebas:
        v = puntuacion(E.lobo(pares, agg, dev, c)); res[n] = v
        print(f"{n:<42}{v:>10.5f}", flush=True)
    m = min(res, key=res.get)
    print(f"\nMEJOR EN DESARROLLO: {m}  ALC={res[m]:.5f}")
    json.dump({"resultados": res, "mejor": m}, open("barrido_prior.json", "w"), indent=1)

    # Este barrido ES la fuente de la fuerza entregada: escribe hiper.json con el
    # argmin de desarrollo entre las variantes beta-binomiales. Antes se escribía
    # a mano, que es justo lo que la verificación independiente marcó como no
    # trazable: el pipeline publicado no reproducía S=3.
    bb = {int(k.split("=")[1]): v for k, v in res.items() if k.startswith("beta_binomial S=")}
    s = min(bb, key=bb.get)
    plano = ", ".join(f"S={k}: {bb[k]:.5f}" for k in sorted(bb))
    json.dump({"fuerza": s,
               "mejor_en_dev": f"beta_binomial p0=0.5, argmin de desarrollo ({plano})",
               "generado_por": "barrido_prior.py"},
              open("hiper.json", "w"), indent=1)
    print(f"hiper.json <- fuerza = {s} (argmin de desarrollo; {plano})")


if __name__ == "__main__":
    main()
