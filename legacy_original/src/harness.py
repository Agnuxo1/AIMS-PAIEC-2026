"""Simulación local de la evaluación oficial del PAIEC (Brier ALC).

Reproduce la DEFINICIÓN PUBLICADA en https://aims.stanford.edu/competition.
No afirma igualdad numérica con la puntuación oficial: no tenemos ni el evaluador
ni los datos ocultos. Lo que sí se comprueba (tests/test_metrica.py) es que la
implementación cumple la definición en casos construidos a mano.

Reglas implementadas:
  - Un par = (subject_id, benchmark_id). Solo pares con >= 80 ítems distintos.
  - Los ítems del par se parten 50/50 en pool de adquisición y de evaluación,
    con semilla fija: la partición no cambia entre presupuestos ni entre métodos.
  - Los ítems de adquisición nunca se puntúan; los de evaluación nunca se revelan.
  - Presupuestos 0, 1, 3, 7, 15, 31 etiquetas por par, con conjuntos ANIDADOS
    (las de un presupuesto menor siguen disponibles en el mayor).
  - Política por defecto: los candidatos llegan en orden aleatorio fijo y se
    revelan los n primeros (equivale a la selección aleatoria de la plataforma).
  - predict(input, labeled) se llama una vez por [subject, item] DISTINTO del
    pool de evaluación; las respuestas repetidas del mismo par reutilizan esa
    predicción y todas cuentan en el Brier.
  - Brier ALC del par = 0.1*B0 + 0.2*B1 + 0.2*B3 + 0.2*B7 + 0.2*B15 + 0.1*B31
  - Puntuación final = media SIMPLE de los ALC de los pares (cada par pesa igual,
    independientemente de cuántas respuestas tenga).
"""
from __future__ import annotations

import glob
import hashlib
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

DATA = r"E:\AIMS-sprint\data"
BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)
MIN_ITEMS = 80
MAX_LABELS = 31


# --------------------------------------------------------------------------- datos

def benchmarks_binarios(catalogo: str = r"E:\AIMS-sprint\meta\catalogo_benchmarks.parquet") -> pd.DataFrame:
    cat = pd.read_parquet(catalogo)
    return cat[(cat["release"] == "public") & (cat["n_response_values"] == 2)].copy()


def carga_respuestas(benchmark: str) -> pd.DataFrame | None:
    """Respuestas de un benchmark, solo columnas necesarias, etiqueta binaria."""
    f = os.path.join(DATA, benchmark, "formatted_tables", "responses.parquet")
    if not os.path.exists(f):
        return None
    d = pd.read_parquet(f, columns=["subject_id", "item_id", "benchmark_id", "response"])
    d = d.dropna(subset=["response"])
    vals = pd.unique(d["response"])
    if not set(np.unique(vals)).issubset({0.0, 1.0, 0, 1}):
        return None  # no es binario de verdad: fuera
    d["y"] = d["response"].astype(np.int8)
    return d.drop(columns=["response"])


def carga_sujetos(benchmark: str) -> pd.DataFrame:
    f = os.path.join(DATA, benchmark, "formatted_tables", "subjects.parquet")
    return pd.read_parquet(f) if os.path.exists(f) else pd.DataFrame()


def semilla_estable(*partes) -> int:
    """Semilla derivada con hashlib, NO con hash().

    `hash()` de Python se aleatoriza por proceso (PYTHONHASHSEED), asi que una
    particion derivada de el NO es reproducible entre ejecuciones. Lo detecto la
    verificacion independiente de este sprint. blake2b da el mismo valor siempre.
    """
    h = hashlib.blake2b(digest_size=8)
    for x in partes:
        h.update(str(x).encode("utf-8")); h.update(b"|")
    return int.from_bytes(h.digest(), "big") % (2**32)


# --------------------------------------------------------------------------- pares

@dataclass
class Par:
    """Un par sujeto-benchmark listo para evaluar."""
    subject_id: str
    benchmark_id: str
    adq_items: np.ndarray        # ítems del pool de adquisición (orden de llegada fijo)
    adq_y: np.ndarray            # etiqueta de cada ítem de adquisición (1 por ítem)
    eva_items: np.ndarray        # ítems distintos del pool de evaluación
    eva_y: np.ndarray            # TODAS las respuestas de evaluación
    eva_idx: np.ndarray          # a qué ítem de eva_items pertenece cada respuesta


def construye_pares(d: pd.DataFrame, semilla: int = 20261024, max_pares: int | None = None) -> list[Par]:
    """Parte cada par 50/50 y fija el orden de adquisición. Semilla fija."""
    pares: list[Par] = []
    rng = np.random.default_rng(semilla)
    for (sid, bid), g in d.groupby(["subject_id", "benchmark_id"], sort=True):
        items = pd.unique(g["item_id"])
        if len(items) < MIN_ITEMS:
            continue
        # partición determinista por par: la semilla depende del par, no del orden
        r = np.random.default_rng(semilla_estable(semilla, sid, bid))
        perm = r.permutation(len(items))
        corte = len(items) // 2
        adq_ids, eva_ids = items[perm[:corte]], items[perm[corte:]]

        # etiqueta por ítem de adquisición: una respuesta registrada (la primera)
        prim = g.drop_duplicates("item_id").set_index("item_id")["y"]
        adq_y = prim.reindex(adq_ids).to_numpy(dtype=np.int8)

        ge = g[g["item_id"].isin(set(eva_ids))]
        eva_ids = pd.unique(ge["item_id"])
        pos = {it: i for i, it in enumerate(eva_ids)}
        pares.append(Par(sid, bid, adq_ids, adq_y, eva_ids,
                         ge["y"].to_numpy(dtype=np.int8),
                         ge["item_id"].map(pos).to_numpy(dtype=np.int32)))
    if max_pares is not None and len(pares) > max_pares:
        sel = rng.choice(len(pares), max_pares, replace=False)
        pares = [pares[i] for i in sorted(sel)]
    return pares


# --------------------------------------------------------------------------- métrica

def brier_alc_par(p_por_presupuesto: dict[int, np.ndarray], par: Par) -> tuple[float, dict[int, float]]:
    """ALC de un par. p_por_presupuesto[n] = predicción por ítem DISTINTO de evaluación."""
    por_b = {}
    for n in BUDGETS:
        p = np.asarray(p_por_presupuesto[n], dtype=np.float64)[par.eva_idx]
        por_b[n] = float(np.mean((p - par.eva_y) ** 2))
    alc = float(sum(w * por_b[n] for n, w in zip(BUDGETS, WEIGHTS)))
    return alc, por_b


def evalua(pares: list[Par], predictor) -> pd.DataFrame:
    """Ejecuta un predictor sobre todos los pares. Devuelve una fila por par.

    `predictor` es una función predice(par, labeled_idx) -> array de p por ítem
    distinto de evaluación. `labeled_idx` son los índices revelados del pool de
    adquisición (anidados y crecientes).
    """
    filas = []
    for par in pares:
        pb = {}
        for n in BUDGETS:
            k = min(n, len(par.adq_items))
            pb[n] = predictor(par, np.arange(k))
        alc, por_b = brier_alc_par(pb, par)
        filas.append({"subject_id": par.subject_id, "benchmark_id": par.benchmark_id,
                      "n_items_eval": len(par.eva_items), "n_resp_eval": len(par.eva_y),
                      "brier_alc": alc, **{f"B{n}": por_b[n] for n in BUDGETS}})
    return pd.DataFrame(filas)


def puntuacion(res: pd.DataFrame) -> float:
    """Media simple de los ALC de los pares: cada par pesa igual."""
    return float(res["brier_alc"].mean())
