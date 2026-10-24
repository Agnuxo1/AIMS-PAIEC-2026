"""Casos construidos a mano con resultado conocido para el simulador de la métrica.

No prueban igualdad con la puntuación oficial (no tenemos el evaluador): prueban
que nuestra implementación cumple la definición publicada. Ejecutar:
    python work/test_metrica.py
"""
import numpy as np

from harness import BUDGETS, WEIGHTS, Par, brier_alc_par, evalua, puntuacion
from predictores import BetaBinomial, constante_mitad, media_empirica

fallos = []


def check(nombre, obtenido, esperado, tol=1e-9):
    ok = abs(obtenido - esperado) <= tol
    print(f"  [{'OK ' if ok else 'MAL'}] {nombre}: obtenido={obtenido:.8f} esperado={esperado:.8f}")
    if not ok:
        fallos.append(nombre)


def par_simple(n_items=100, y_eva=None, adq_y=None):
    eva_items = np.arange(n_items)
    if y_eva is None:
        y_eva = np.zeros(n_items, dtype=np.int8)
    adq = np.arange(n_items)
    if adq_y is None:
        adq_y = np.ones(n_items, dtype=np.int8)
    return Par("s", "b", adq, adq_y, eva_items, np.asarray(y_eva, dtype=np.int8),
               np.arange(len(y_eva), dtype=np.int32))


print("1. Pesos y presupuestos coinciden con las bases")
check("suma de pesos", sum(WEIGHTS), 1.0)
check("n presupuestos", len(BUDGETS), 6)
print(f"  presupuestos={BUDGETS} pesos={WEIGHTS}")

print("\n2. Predecir siempre 0.5 da Brier 0.25 en todos los presupuestos")
p = par_simple(y_eva=np.array([0, 1] * 50))
pb = {n: constante_mitad(p, np.arange(min(n, 100))) for n in BUDGETS}
alc, por_b = brier_alc_par(pb, p)
check("ALC de la constante 0.5", alc, 0.25)
check("B0 de la constante 0.5", por_b[0], 0.25)

print("\n3. Predicción perfecta da ALC 0")
pb = {n: p.eva_y.astype(float) for n in BUDGETS}
alc, _ = brier_alc_par(pb, p)
check("ALC perfecto", alc, 0.0)

print("\n4. El baseline oficial con n=1 predice 0 o 1 y un fallo cuesta 1.0")
# adquisición todo unos, evaluación todo ceros -> con 1 etiqueta predice 1.0
p = par_simple(y_eva=np.zeros(100, dtype=np.int8), adq_y=np.ones(100, dtype=np.int8))
b1 = media_empirica(p, np.arange(1))
check("prediccion del baseline con n=1", float(b1[0]), 1.0)
pb = {n: media_empirica(p, np.arange(min(n, 100))) for n in BUDGETS}
alc, por_b = brier_alc_par(pb, p)
check("B0 del baseline (sin etiquetas -> 0.5)", por_b[0], 0.25)
check("B1 del baseline (fallo total)", por_b[1], 1.0)
check("ALC del baseline en este caso", alc, 0.1 * 0.25 + 0.9 * 1.0)
print("  -> es el modo de fallo que el informe debe explicar: 0.925 frente a 0.25 de no hacer nada")

print("\n5. La contraccion beta-binomial evita ese modo de fallo")
bb = BetaBinomial(p0=0.5, fuerza=8.0)
pb = {n: bb(p, np.arange(min(n, 100))) for n in BUDGETS}
alc_bb, por_bb = brier_alc_par(pb, p)
check("B1 con fuerza 8", por_bb[1], ((0.5 * 8 + 1) / (8 + 1)) ** 2)
print(f"  ALC beta-binomial={alc_bb:.4f} frente a baseline={alc:.4f}")
if alc_bb >= alc:
    fallos.append("la contraccion deberia mejorar en este caso")

print("\n6. Cada par pesa igual, sin importar cuantas respuestas tenga")
import pandas as pd
res = pd.DataFrame({"brier_alc": [0.0, 1.0], "n_resp_eval": [1, 1_000_000]})
check("media simple de pares", puntuacion(res), 0.5)

print("\n7. Las respuestas repetidas del mismo item reutilizan la prediccion")
# dos respuestas del mismo item: una 1 y otra 0, prediccion 0.5 -> Brier 0.25
par = Par("s", "b", np.arange(100), np.ones(100, dtype=np.int8), np.array([0]),
          np.array([1, 0], dtype=np.int8), np.array([0, 0], dtype=np.int32))
pb = {n: np.array([0.5]) for n in BUDGETS}
alc, _ = brier_alc_par(pb, par)
check("respuestas repetidas", alc, 0.25)

print("\n8. Conjuntos de etiquetas anidados: n menor es prefijo de n mayor")
anidado = all(set(range(a)) <= set(range(b)) for a, b in zip(BUDGETS, BUDGETS[1:]))
print(f"  [{'OK ' if anidado else 'MAL'}] prefijos anidados: {anidado}")
if not anidado:
    fallos.append("anidamiento")

print("\n9. El pool de adquisicion y el de evaluacion no se solapan")
from harness import construye_pares
d = pd.DataFrame({"subject_id": ["s"] * 200, "benchmark_id": ["b"] * 200,
                  "item_id": [f"i{i}" for i in range(200)],
                  "y": np.random.default_rng(0).integers(0, 2, 200).astype(np.int8)})
pares = construye_pares(d)
solape = set(pares[0].adq_items) & set(pares[0].eva_items)
print(f"  [{'OK ' if not solape else 'MAL'}] solape = {len(solape)} items; "
      f"adq={len(pares[0].adq_items)} eva={len(pares[0].eva_items)}")
if solape:
    fallos.append("solape adquisicion/evaluacion")

print("\n10. Se descartan los pares con menos de 80 items distintos")
d2 = d.head(79)
print(f"  [{'OK ' if not construye_pares(d2) else 'MAL'}] 79 items -> {len(construye_pares(d2))} pares")
if construye_pares(d2):
    fallos.append("umbral de 80 items")

print("\n" + "=" * 60)
print(f"RESULTADO: {'TODOS LOS CASOS PASAN' if not fallos else 'FALLOS: ' + ', '.join(fallos)}")
raise SystemExit(1 if fallos else 0)
