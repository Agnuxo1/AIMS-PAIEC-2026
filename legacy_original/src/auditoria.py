"""Auditoría adversaria ACOTADA del release público de measurement-db.

Alcance deliberadamente limitado (ver PLAN.md 2.F): cribado barato de esquema,
duplicados EXACTOS y campos entregados al predictor. Cada resultado se etiqueta
CONFIRMADO (cuantificado, con ejemplos verificables) o INDICIO SIN CONFIRMAR.

LÍNEA ROJA respetada: las bases prohíben intentar identificar qué sujeto o
benchmark hay tras los IDs anónimos del feedback formativo. Esta auditoría solo
toca datos PÚBLICOS del release de entrenamiento y no usa feedback de la
plataforma para nada.

Definiciones fijadas ANTES de mirar los datos:
  - "mismo ítem": igual `content_hash` tal y como lo publica el propio dataset.
  - "repetición": dos filas de responses con el mismo (subject_id, item_id).
  - "campo posterior a la respuesta": columna de items que describe cómo se
    corrige (`grading_criterion`, `verifier`) y no lo que se le muestra al
    sistema (`content`).
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
import pandas as pd

DATA = r"E:\AIMS-sprint\data"
SALIDA = r"E:\AIMS-sprint\work\auditoria.json"

# Lo que el predictor recibe de verdad, segun las bases:
#   subject: normalized_name, provider, release_date, access_date, harness,
#            harness_version, reasoning_effort, subject_features_extra
#   item:    item_content, item_features, interactors, benchmark_id
CAMPOS_AL_PREDICTOR = {"content", "item_features", "interactors", "benchmark_id"}
CAMPOS_NO_ENTREGADOS = {"grading_criterion", "verifier", "content_hash", "raw_item_id",
                        "asset_manifest", "item_id"}

PISTA = re.compile(r"\b(answer|solution|correct|gold|label|target|ground[_ ]?truth|"
                   r"respuesta|solucion|etiqueta)\b", re.I)


def benchs():
    cat = pd.read_parquet(r"E:\AIMS-sprint\meta\catalogo_benchmarks.parquet")
    return sorted(cat[(cat["release"] == "public") & (cat["n_response_values"] == 2)]["benchmark_id"])


def a1_duplicados(nombres):
    """Ítems idénticos (mismo content_hash) que aparecen en más de un benchmark."""
    vistos: dict[str, set] = {}
    total = 0
    for b in nombres:
        f = os.path.join(DATA, b, "formatted_tables", "items.parquet")
        if not os.path.exists(f):
            continue
        d = pd.read_parquet(f, columns=["content_hash"]).dropna()
        total += len(d)
        for h in pd.unique(d["content_hash"]):
            vistos.setdefault(h, set()).add(b)
    cruza = {h: bs for h, bs in vistos.items() if len(bs) > 1}
    pares = {}
    for bs in cruza.values():
        for x in sorted(bs):
            for y in sorted(bs):
                if x < y:
                    pares[(x, y)] = pares.get((x, y), 0) + 1
    top = sorted(pares.items(), key=lambda kv: -kv[1])[:15]
    return {
        "veredicto": "CONFIRMADO" if cruza else "sin hallazgo",
        "items_totales": int(total),
        "hashes_distintos": len(vistos),
        "hashes_en_mas_de_un_benchmark": len(cruza),
        "porcentaje_de_hashes_compartidos": round(100 * len(cruza) / max(1, len(vistos)), 4),
        "benchmarks_implicados": len({b for bs in cruza.values() for b in bs}),
        "pares_de_benchmarks_con_mas_solape": [
            {"a": a, "b": c, "items_compartidos": int(n)} for (a, c), n in top],
        "nota": ("Mide solapamiento DENTRO del release publico de entrenamiento. "
                 "No dice nada sobre la frontera con el conjunto privado de test, "
                 "que no tenemos y que no se puede auditar desde aqui."),
    }


def a2_fuga_en_campos(nombres):
    """¿Algún campo que SÍ llega al predictor contiene la solución o el criterio?"""
    sospechas, revisados, con_feat = [], 0, 0
    ejemplos = []
    for b in nombres[:60]:          # cribado acotado
        f = os.path.join(DATA, b, "formatted_tables", "items.parquet")
        if not os.path.exists(f):
            continue
        d = pd.read_parquet(f)
        revisados += 1
        if "item_features" not in d.columns:
            continue
        s = d["item_features"].dropna().astype(str)
        if s.empty:
            continue
        con_feat += 1
        m = s[s.str.contains(PISTA, na=False)]
        if len(m):
            # ¿coincide con el criterio de correccion? eso seria lo grave
            coincide = False
            if "grading_criterion" in d.columns:
                gc = d["grading_criterion"].dropna().astype(str)
                if len(gc):
                    comun = set(s.head(500)) & set(gc.head(500))
                    coincide = len(comun) > 0
            sospechas.append({"benchmark": b, "filas_con_palabra_clave": int(len(m)),
                              "de": int(len(s)),
                              "item_features_igual_a_grading_criterion": bool(coincide)})
            if len(ejemplos) < 5:
                ejemplos.append({"benchmark": b, "muestra": m.iloc[0][:180]})
    graves = [x for x in sospechas if x["item_features_igual_a_grading_criterion"]]
    return {
        "veredicto": "CONFIRMADO" if graves else ("INDICIO SIN CONFIRMAR" if sospechas else "sin hallazgo"),
        "benchmarks_revisados": revisados,
        "con_item_features_no_vacio": con_feat,
        "benchmarks_con_palabra_clave_en_item_features": len(sospechas),
        "benchmarks_donde_item_features_coincide_con_grading_criterion": len(graves),
        "detalle": sospechas[:15],
        "ejemplos": ejemplos,
        "nota": ("Una palabra como 'answer' en item_features NO es fuga por si sola: "
                 "puede describir el formato de respuesta esperado. Solo se marca "
                 "CONFIRMADO si el campo entregado al predictor coincide con el "
                 "criterio de correccion, que el predictor no deberia ver."),
        "campos_no_entregados_al_predictor": sorted(CAMPOS_NO_ENTREGADOS),
    }


def a3_etiquetas_fragiles(nombres):
    """Repeticiones del mismo (sujeto, ítem) que discrepan entre sí."""
    filas = []
    for b in nombres:
        f = os.path.join(DATA, b, "formatted_tables", "responses.parquet")
        if not os.path.exists(f):
            continue
        d = pd.read_parquet(f, columns=["subject_id", "item_id", "response"]).dropna()
        if d.empty:
            continue
        g = d.groupby(["subject_id", "item_id"])["response"].agg(["count", "mean"])
        rep = g[g["count"] > 1]
        if rep.empty:
            continue
        discrepa = ((rep["mean"] > 0) & (rep["mean"] < 1)).sum()
        filas.append({"benchmark": b, "pares_repetidos": int(len(rep)),
                      "pares_que_discrepan": int(discrepa),
                      "tasa": round(float(discrepa / len(rep)), 4),
                      "repeticiones_max": int(rep["count"].max())})
    t = pd.DataFrame(filas)
    if t.empty:
        return {"veredicto": "sin hallazgo", "benchmarks_con_repeticiones": 0}
    peores = t.sort_values("tasa", ascending=False).head(12)
    return {
        "veredicto": "CONFIRMADO",
        "benchmarks_con_repeticiones": int(len(t)),
        "pares_repetidos_totales": int(t["pares_repetidos"].sum()),
        "pares_que_discrepan_totales": int(t["pares_que_discrepan"].sum()),
        "tasa_global_de_discrepancia": round(float(t["pares_que_discrepan"].sum() /
                                                   max(1, t["pares_repetidos"].sum())), 4),
        "tasa_mediana_por_benchmark": round(float(t["tasa"].median()), 4),
        "peores": peores.to_dict("records"),
        "nota": ("Una repeticion discordante NO prueba un fallo del corrector: puede "
                 "ser muestreo estocastico del propio sistema. Lo relevante para el "
                 "concurso es que fija un TECHO a la predictibilidad: ninguna "
                 "prediccion puede acertar ambas respuestas de un par discordante."),
    }


def a4_techo_de_predictibilidad(nombres):
    """Brier mínimo alcanzable: predecir la tasa verdadera de cada par.

    Da el suelo irreducible por ruido de etiqueta, que contextualiza cualquier
    mejora: si el suelo es 0.18, bajar de 0.24 a 0.21 es la mitad del recorrido.
    """
    # pickle: cache local propia generada por construye.py en esta misma maquina
    import pickle
    with open(r"E:\AIMS-sprint\work\pares.pkl", "rb") as f:
        pares = pickle.load(f)
    suelos = []
    for p in pares:
        y = p.eva_y
        if len(y) == 0:
            continue
        # oraculo por item: la media de las respuestas de ESE item
        s = pd.Series(y).groupby(p.eva_idx).transform("mean").to_numpy()
        suelos.append(float(np.mean((s - y) ** 2)))
    orac_par = []
    for p in pares:
        if len(p.eva_y) == 0:
            continue
        m = p.eva_y.mean()
        orac_par.append(float(np.mean((m - p.eva_y) ** 2)))
    return {
        "veredicto": "CONFIRMADO",
        "conjunto": ("LOS 769 PARES del conjunto de validacion completo, no los 496 "
                     "de cabecera. Las cifras que acompanan a la tabla principal se "
                     "calculan sobre cabecera en sensibilidad.py; no mezclar."),
        "brier_oraculo_por_item": round(float(np.mean(suelos)), 5),
        "brier_oraculo_por_par": round(float(np.mean(orac_par)), 5),
        "nota": ("Oraculo por item: conoce la tasa verdadera de cada item (suelo "
                 "absoluto). Oraculo por par: conoce la tasa del par pero predice "
                 "lo mismo para todos sus items, que es lo maximo que puede lograr "
                 "cualquier predictor constante por par como el nuestro y como el "
                 "baseline oficial."),
    }


def main():
    nombres = benchs()
    print(f"benchmarks binarios publicos: {len(nombres)}\n")
    r = {}
    for clave, fn in (("A_duplicados_entre_benchmarks", a1_duplicados),
                      ("B_fuga_en_campos_entregados", a2_fuga_en_campos),
                      ("C_etiquetas_fragiles", a3_etiquetas_fragiles),
                      ("D_techo_de_predictibilidad", a4_techo_de_predictibilidad)):
        print(f"-> {clave}", flush=True)
        r[clave] = fn(nombres)
        print(json.dumps(r[clave], ensure_ascii=False, indent=1)[:1100], "\n", flush=True)
    json.dump(r, open(SALIDA, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("guardado", SALIDA)


if __name__ == "__main__":
    main()
