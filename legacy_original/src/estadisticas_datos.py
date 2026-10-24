"""Estadisticas descriptivas que cita el informe, con su script propio.

La verificacion independiente senalo que varias cifras del informe (la media
macro por par, la fraccion de pares bajo 0.05, la moda de la distribucion y la
dispersion del sujeto frente a la dispersion entre pares) no las producia ningun
script. Aqui estan, y se distingue entre TODAS las filas sujeto-benchmark y las
ELEGIBLES (>=80 respuestas), que son las unicas que el concurso evalua.
"""
import json
import numpy as np
import pandas as pd

W = r"E:\AIMS-sprint\work"

def descripcion(t, etiqueta):
    p = (t["k"] / t["n"].clip(lower=1)).to_numpy()
    h, bordes = np.histogram(p, bins=20, range=(0, 1))
    i = int(np.argmax(h))
    return {"etiqueta": etiqueta, "n_filas": int(len(t)),
            "media_macro_por_par": round(float(p.mean()), 4),
            "media_ponderada_por_respuesta": round(float(t["k"].sum() / t["n"].sum()), 4),
            "sd_entre_pares": round(float(p.std()), 4),
            "fraccion_bajo_0.05": round(float(np.mean(p < 0.05)), 4),
            "fraccion_sobre_0.95": round(float(np.mean(p > 0.95)), 4),
            "moda_intervalo": [round(float(bordes[i]), 2), round(float(bordes[i + 1]), 2)],
            "histograma": h.tolist()}

def main():
    agg = pd.read_parquet(rf"{W}\agregados.parquet")
    out = {"todas": descripcion(agg, "todas las filas sujeto-benchmark"),
           "elegibles": descripcion(agg[agg["n"] >= 80], "filas con >=80 respuestas")}
    for t, clave in ((agg, "todas"), (agg[agg["n"] >= 80], "elegibles")):
        t = t.copy(); t["p"] = t["k"] / t["n"].clip(lower=1)
        v = t.groupby("subject_id")["p"].agg(["mean", "std", "count"])
        v5 = v[v["count"] >= 5]
        out[clave]["sujetos_con_5_o_mas_benchmarks"] = int(len(v5))
        out[clave]["sd_media_del_sujeto_entre_benchmarks"] = round(float(v5["std"].mean()), 4)
    for k in ("todas", "elegibles"):
        d = out[k]
        print(f"=== {d['etiqueta']} ({d['n_filas']} filas)")
        print(f"   media macro por par = {d['media_macro_por_par']}  "
              f"(ponderada por respuesta = {d['media_ponderada_por_respuesta']})")
        print(f"   sd entre pares = {d['sd_entre_pares']}  bajo 0.05 = {100*d['fraccion_bajo_0.05']:.1f} %  "
              f"sobre 0.95 = {100*d['fraccion_sobre_0.95']:.1f} %")
        print(f"   moda en el intervalo {d['moda_intervalo']}")
        print(f"   sujetos con >=5 benchmarks = {d['sujetos_con_5_o_mas_benchmarks']}, "
              f"sd media de su tasa entre benchmarks = {d['sd_media_del_sujeto_entre_benchmarks']}")
    json.dump(out, open(rf"{W}\estadisticas_datos.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\nguardado estadisticas_datos.json")

if __name__ == "__main__":
    main()
