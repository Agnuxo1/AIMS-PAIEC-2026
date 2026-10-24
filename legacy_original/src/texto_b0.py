"""¿Hay señal transferible en el texto del ítem para el presupuesto 0?

Es la objeción más fuerte de la ronda 2: nuestro B0 es exactamente 0.25 porque
predecimos 0.5 sin etiquetas. Si rasgos baratos del enunciado (longitud, código,
opciones, idioma, LaTeX) más atributos del sujeto predicen algo en un benchmark
NUEVO, la tesis "no hay señal transferible" se cae.

Protocolo: se entrena y se elige SOLO con los 50 benchmarks de desarrollo, con
validación LOBO dentro de desarrollo. La cabecera no se toca aquí.
Nada de modelos grandes ni de embeddings: rasgos calculables dentro de predict().
"""
from __future__ import annotations

import os
import re
import warnings

import numpy as np
import pandas as pd

import ejecuta as E

warnings.filterwarnings("ignore")
DATA = r"E:\AIMS-sprint\data"
MAX_FILAS_POR_BENCH = 40_000

RE_CODIGO = re.compile(r"```|\bdef \b|\bclass \b|#include|\bimport \b|;\s*\n|\{\s*\n")
RE_OPCION = re.compile(r"(?m)^\s*\(?[A-Ea-e][\).]\s+")
RE_LATEX = re.compile(r"\$|\\frac|\\begin\{|\\sum|\\int")


def rasgos_texto(s: pd.Series) -> pd.DataFrame:
    s = s.fillna("").astype(str)
    n = s.str.len().clip(lower=1)
    f = pd.DataFrame(index=s.index)
    f["log_long"] = np.log1p(s.str.len())
    f["log_palabras"] = np.log1p(s.str.count(r"\s+"))
    f["prop_digitos"] = s.str.count(r"\d") / n
    f["prop_mayus"] = s.str.count(r"[A-Z]") / n
    f["prop_no_ascii"] = s.str.count(r"[^\x00-\x7F]") / n
    f["prop_puntuacion"] = s.str.count(r"[^\w\s]") / n
    f["tiene_codigo"] = s.str.contains(RE_CODIGO, regex=True).astype(float)
    f["tiene_latex"] = s.str.contains(RE_LATEX, regex=True).astype(float)
    f["n_opciones"] = s.str.count(RE_OPCION).clip(upper=8)
    f["es_opcion_multiple"] = (f["n_opciones"] >= 3).astype(float)
    f["interrogaciones"] = s.str.count(r"\?").clip(upper=5)
    f["saltos_linea"] = np.log1p(s.str.count("\n"))
    return f


def rasgos_sujeto(sub: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=sub.index)
    ry = pd.to_datetime(sub.get("release_date"), errors="coerce")
    f["anio_release"] = ry.dt.year.fillna(2024).clip(2020, 2027) - 2024
    f["tiene_esfuerzo"] = sub.get("reasoning_effort").notna().astype(float) \
        if "reasoning_effort" in sub else 0.0
    f["tiene_harness"] = sub.get("harness").notna().astype(float) if "harness" in sub else 0.0
    prov = sub.get("provider").fillna("?").astype(str) if "provider" in sub else pd.Series("?", index=sub.index)
    for p in ("OpenAI", "Anthropic", "Google", "Meta", "Mistral", "DeepSeek", "Alibaba", "xAI"):
        f[f"prov_{p}"] = (prov == p).astype(float)
    return f


def carga_dev(benches):
    trozos = []
    for b in sorted(benches):
        fi = os.path.join(DATA, b, "formatted_tables", "items.parquet")
        fr = os.path.join(DATA, b, "formatted_tables", "responses.parquet")
        fs = os.path.join(DATA, b, "formatted_tables", "subjects.parquet")
        if not all(os.path.exists(x) for x in (fi, fr, fs)):
            continue
        r = pd.read_parquet(fr, columns=["subject_id", "item_id", "benchmark_id", "response"]).dropna()
        if r.empty:
            continue
        if len(r) > MAX_FILAS_POR_BENCH:
            r = r.sample(MAX_FILAS_POR_BENCH, random_state=0)
        it = pd.read_parquet(fi, columns=["item_id", "content"])
        su = pd.read_parquet(fs)
        d = r.merge(it, on="item_id", how="left").merge(su, on="subject_id", how="left")
        if d.empty:
            continue
        X = pd.concat([rasgos_texto(d["content"]), rasgos_sujeto(d)], axis=1)
        X["benchmark_id"] = d["benchmark_id"].values
        X["y"] = d["response"].astype(int).values
        trozos.append(X)
    return pd.concat(trozos, ignore_index=True) if trozos else pd.DataFrame()


def main():
    pares, agg, dev, cab = E.carga()
    print(f"cargando {len(dev)} benchmarks de desarrollo...", flush=True)
    D = carga_dev(dev)
    print(f"filas={len(D):,}  benchmarks={D['benchmark_id'].nunique()}  tasa de acierto={D['y'].mean():.4f}\n")

    cols = [c for c in D.columns if c not in ("benchmark_id", "y")]
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    benches = sorted(D["benchmark_id"].unique())
    brier_modelo, brier_const, brier_mediaglobal = [], [], []
    for b in benches:
        tr, te = D[D["benchmark_id"] != b], D[D["benchmark_id"] == b]
        if len(te) < 200 or len(tr) < 5000:
            continue
        sc = StandardScaler().fit(tr[cols])
        m = LogisticRegression(max_iter=600, C=0.1).fit(sc.transform(tr[cols]), tr["y"])
        p = m.predict_proba(sc.transform(te[cols]))[:, 1]
        y = te["y"].to_numpy()
        brier_modelo.append(np.mean((p - y) ** 2))
        brier_const.append(np.mean((0.5 - y) ** 2))
        brier_mediaglobal.append(np.mean((tr["y"].mean() - y) ** 2))

    bm, bc, bg = map(np.mean, (brier_modelo, brier_const, brier_mediaglobal))
    gana = int(np.sum(np.array(brier_modelo) < np.array(brier_const)))
    n = len(brier_modelo)
    from math import comb
    k = min(gana, n - gana)
    pval = min(1.0, sum(comb(n, i) for i in range(k + 1)) * 2 / 2 ** n)
    print(f"B0 con validacion LOBO DENTRO de desarrollo ({n} benchmarks):")
    print(f"  constante 0.5               Brier = {bc:.5f}")
    print(f"  media global de entrenamiento Brier = {bg:.5f}")
    print(f"  regresion sobre rasgos      Brier = {bm:.5f}")
    print(f"  gana al 0.5 en {gana}/{n} benchmarks, p={pval:.4f}")
    print(f"  diferencia media = {bm-bc:+.5f}")

    print("\nRasgos con mayor coeficiente (modelo ajustado con todo desarrollo):")
    sc = StandardScaler().fit(D[cols])
    m = LogisticRegression(max_iter=600, C=0.1).fit(sc.transform(D[cols]), D["y"])
    imp = pd.Series(m.coef_[0], index=cols).sort_values(key=abs, ascending=False)
    for k2, v in imp.head(10).items():
        print(f"   {k2:<22} {v:+.4f}")

    efecto_alc = 0.1 * (bm - bc)
    print(f"\nEfecto maximo sobre el Brier ALC (B0 pesa 0.1): {efecto_alc:+.5f}")
    veredicto = ("HAY senal transferible pequena" if bm < bc and pval < 0.05
                 else "NO se demuestra senal transferible")
    print(f"VEREDICTO: {veredicto}")
    import json
    json.dump({"n_benchmarks": n, "brier_constante": float(bc), "brier_media_global": float(bg),
               "brier_modelo": float(bm), "gana": gana, "p": float(pval),
               "efecto_sobre_alc": float(efecto_alc), "veredicto": veredicto,
               "coeficientes": imp.head(12).to_dict()},
              open(r"E:\AIMS-sprint\work\texto_b0.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
