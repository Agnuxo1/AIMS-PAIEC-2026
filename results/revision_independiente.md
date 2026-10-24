# Revisión independiente del sprint original

Fecha: 24-10-2026. Segunda opinión solicitada por el agente principal. Alcance: lectura de los 15 scripts de `entrega/repo/src`, el modelo entregado, README, fuente del informe, 14 resultados publicados y scripts/logs pertinentes de `work`, `verificacion`, `verificacion2` y `verificacion3`. Se hicieron cálculos propios sobre JSON/CSV/parquet originales y contrajemplos sintéticos. No se ejecutaron ni importaron módulos del proyecto original, ni se abrió `pares.pkl`. No se leyeron puntuaciones de los experimentos nuevos. Solo se ha escrito este documento.

## Conclusión respaldada

La mejora original sobre la media empírica es coherente en los resultados conservados: 496 pares, 101 benchmarks; diferencia por par −0,03488753, mejora en 86 benchmarks y p exacta de signos 2,8269e−13. Esto comprueba consistencia de artefactos originales, no reconstrucción completa desde datos brutos ni puntuación oficial.

No hay evidencia observable de ocultación de los mejores resultados guardados. El método Empirical Bayes condicionado al sujeto, 0,20989535, aparece primero en README:37 y en informe/informe.tex:175. El jerárquico macro, 0,21024007, aparece en la tabla completa del informe:176 y en resultados_cabecera.csv. Ambos mejoran puntualmente el entregado, 0,21049698. No se puede inferir intención ni reconstruir experimentos borrados/no guardados.

Sí hay defectos metodológicos y formulaciones demasiado amplias que justifican reconstrucción y nuevos experimentos. La elección de un modelo sencillo no prueba que las alternativas estén agotadas.

## Hallazgos priorizados

### 1. ALTA: la cabecera no está aislada del ajuste de modelos ricos

`src/ejecuta.py:44-54` ajusta cada predictor con `agg[agg["benchmark_id"] != b]`, aunque `benches` sea únicamente desarrollo. `agg` contiene todas las respuestas públicas, incluidas las de cabecera; `src/construye.py:58-60` agrega antes de cualquier separación. Por ello los barridos jerárquicos y EB en desarrollo usan etiquetas de benchmarks de cabecera para formar sus priors. En la evaluación final también se ajusta con los otros 100 benchmarks de cabecera. El benchmark objetivo concreto sí se excluye: no es una fuga directa de sus propias respuestas, pero tampoco es un conjunto de prueba globalmente aislado.

Esto contradice la lectura fuerte de “cabecera nunca se mira hasta el final” (`ejecuta.py:3-5`) y “solo desarrollo” cuando se habla del ajuste EB. Afecta las conclusiones comparativas sobre priors ricos. El modelo entregado Beta(1,5;1,5) no tiene `ajusta`, por lo que el S=3 no recibe automáticamente esta contaminación. El informe ya reconoce inspección repetida y exploración posterior (`informe.tex:404-425`), pero no explicita esta dependencia de entrenamiento.

Acción recomendada: entrenamiento/validación/prueba anidados por grupos, aprendiendo priors solo con el entrenamiento permitido en cada fold; conservar separados los resultados antiguos LOBO y los nuevos. No llamar a la reevaluación pública una prueba prospectiva inédita.

### 2. ALTA: respuestas >=80 se confunden con ítems distintos >=80

El evaluador original hace bien la elegibilidad de pares (`harness.py:99-101`). Sin embargo, `construye.py:58-59` conserva únicamente suma y número de respuestas, no número de ítems. `predictores.py:190-201` y `239` usan `n>=80` para seleccionar los priors, afirmando que son los pares elegibles. `estadisticas_datos.py:7,31-32` y `sensibilidad.py:101` repiten esa equivalencia incorrecta. El último produce el peso de 11.657 supuestos pares elegibles de cabecera y la sensibilidad ponderada −0,03229028.

Contrajemplo propio ejecutado: 80 respuestas de solo dos ítems pasan `n>=80` y fallan `nunique(item_id)>=80`. No cabe tomar −0,0323 como validación exacta de la media por todos los pares elegibles. Además reponderar el resultado de hasta seis sujetos por benchmark no recupera los resultados de todos sus sujetos.

Acción recomendada: registrar ambas cuentas y reconstruir todas las parejas elegibles desde las respuestas. El agente principal está haciendo ese inventario; esta revisión evita duplicar el barrido bruto.

### 3. ALTA: la evaluación acotada cambia la población y no cubre toda su sensibilidad

`construye.py:23-24,62-65` limita a seis parejas y 150 ítems de evaluación. `harness.py:118-120` muestrea sujetos dentro de cada benchmark. Así se infraponderan benchmarks con muchos sujetos respecto a una evaluación por todos los pares. `sensibilidad.py:41-58` solo reduce esos límites: nunca ensaya más de seis sujetos ni más de 150 ítems, y conserva la partición original. Por tanto no mide directamente el error de usar esos máximos frente a la población completa.

Contrajemplo propio ejecutado: con diferencias por benchmark [−0,01,+0,10] y 1.000 frente a un par elegible, la media por benchmark es +0,045 y la media por pares −0,0098901; incluso el signo puede cambiar. No demuestra que el signo cambie aquí: demuestra por qué el criterio exige evaluar la población real.

`harness.py:108-110` también fija siempre la primera respuesta de cada ítem adquirido. `robustez.py:41-45` cambia el orden de esos ítems, pero no vuelve a seleccionar repetición ni cambia partición. Su desviación de cinco semillas solo mide ese origen de variación. La reconstrucción nueva deberá separar la primera repetición de una elección aleatoria entre las registradas.

### 4. ALTA: experimentos de texto insuficientes para cerrar “no hay señal” o “solo queda un LLM de pago”

`tfidf_item.py:76-87` utiliza `HashingVectorizer`, sin estimación/aplicación IDF, y un único SGDRegressor con alpha=1e−5, 12 épocas y `tol=None`. Trunca a 2.000 caracteres y limita a 4.000 respuestas por benchmark y 40.000 de entrenamiento por fold (`28-29,46-51,70-73`). Los mensajes/veredicto mantienen “TF-IDF” (`112-117`), aunque el README y el informe describen correctamente hashing.

Su endpoint es Brier por respuestas, promediado por benchmark, usando la media del par calculada con el propio `te` como base (`48,89-105`). No es Brier ALC del predictor desplegable ni el protocolo de evidencia adquirida. La media del mismo conjunto es un oráculo retrospectivo sobre esas filas, no una tasa verdadera conocida. La comparación es válida como diagnóstico acotado de ese predictor de residuos, pero no sustituye una evaluación en el endpoint de concurso.

Solo prueba escalas positivas 0,5 y 1,0; no calibra contracción pequeña, signo, intercepto/centrado por par ni interacciones con atributos del sujeto. El modelo de rasgos manuales prueba λ=0/0,5/1/1,5 y un Ridge fijo (`por_item.py:108,123`). La logística B0 mezcla filas de todos los pares sin filtro de 80 ítems y sin ponderación por par (`texto_b0.py:72-84,102-113`). Por tanto sus negativos no son una búsqueda representativa de todas las alternativas baratas.

Datos conservados: la logística mejora puntualmente B0 de 0,25 a 0,24950338, aunque gane solo 18/50 benchmarks; media y prueba de signos responden preguntas distintas. El texto disperso mejora 14/50 y tiene correlación media −0,01625, máxima +0,19590. Estos datos respaldan “no se validó una mejora con estas configuraciones”, sin concluir que toda representación de texto carece de señal transferible. Tampoco permiten concluir que el siguiente paso requiera una API de pago. El informe limita mejor su conclusión en `informe.tex:427-428`, pero el cierre narrativo del sprint era más fuerte de lo que permiten sus experimentos.

### 5. MEDIA: el “suelo irreducible” es un óptimo empírico retrospectivo

`auditoria.py:185-205` y `sensibilidad.py:65-73` calculan la tasa de cada ítem y su pérdida sobre las mismas respuestas de evaluación. La descomposición algebraica de 0,17892112 = 0,00590857 + 0,17301255 es correcta para esas observaciones. No identifica sin supuestos las tasas latentes ni el ruido futuro irreducible. Con una única observación por ítem, el oráculo empírico por ítem da pérdida cero incluso si el proceso real es Bernoulli(0,5), cuya pérdida mínima esperada futura es 0,25.

En consecuencia, “96,7 % de variación entre ítems” describe esa descomposición empírica, no demuestra que el 96,7 % sea señal aprendible a partir del texto. La fracción 52,5 % es margen frente al mejor predictor constante retrospectivo de esas filas; debe rotularse así, no como techo físico o dificultad poblacional demostrada.

### 6. MEDIA: pruebas de métrica no prueban la entrega; algunos checks son débiles

`src/test_metrica.py:9-10` importa `harness` y `predictores`; no `submission/model.py`. Su caso de contracción fija fuerza 8 (`63`), mientras la entrega usa 3 (`submission/model.py:36`). No valida filtrado real de sujetos/benchmarks, formato de entrada, pureza ni concordancia de la entrega con el predictor local. Su primer check comprueba suma de pesos y número de presupuestos, no tuplas exactas (`33-36`); el check de anidamiento verifica únicamente una propiedad de rangos (`84-88`).

Hay verificaciones externas adicionales útiles: `verificacion2/edge.py:2-4,39-46` sí carga el ZIP y compara 20.000 ejemplos con la fórmula, y `verificacion/m.py` también carga modelo. Pero no se incluyen en los 15 scripts publicables y esta revisión no las ha ejecutado por la restricción de no ejecutar código original. Sus fuentes no son evidencia actual de que hayan pasado. No tratar “10 casos de métrica” como certificación de toda la entrega o del evaluador oficial.

### 7. MEDIA: incertidumbre de otro orden de adquisición; nombre incorrecto de prueba

`robustez.py:110-111,133-143` calcula el bootstrap únicamente sobre la primera de las cinco semillas, mientras la tabla principal usa otro orden. El README:43-45 y el informe:194-196 lo explican; la afirmación abreviada 0,2454→0,2105 con IC [−0,0373,−0,0249] sin ese matiz juntaría estimando y ejecución distintos. Un bootstrap propio de 10.000 clústeres benchmark sobre las filas originales de la tabla da IC95 [−0,04106554,−0,02854009], semilla47. La señal permanece favorable, pero no son el mismo intervalo.

`ejecuta.py:71-87` implementa una prueba binomial exacta de signos (cuenta diferencias positivas/negativas); no una permutación de signos de las magnitudes. Los textos “sign-flip” son imprecisos. `robustez.py:76` retorna p_boot=0 si no hay réplicas del otro signo con solo 2.000 remuestreos; eso no demuestra una probabilidad exacta nula. El informe usa acertadamente “no replicate of the opposite sign”.

El bootstrap agrupa por benchmark pero los 11 benchmarks con contenidos compartidos pueden seguir siendo dependientes (`auditoria.json`, descrito por informe:363-370). Para modelos de texto conviene agrupar componentes de contenido y cuantificar tamaños; no basta el LOBO aislado para eliminar todo solapamiento.

### 8. BAJA: reproducción depende de rutas absolutas, revisión de datos no fijada y logs históricos

`README.md:113-114` da hashes de revisión, pero `src/descarga_datos.py:11` no pasa `revision=` a hf_hub_download. Una descarga futura podría cambiar. Los scripts usan rutas absolutas y mezclan escritura relativa/absoluta de hiper/resultados; README:140-143 lo reconoce. `work/rehacer.sh:7` sobrescribe fuerza=4 después del barrido, y `work/rehacer.log:58-80` corresponde a S=4, no al artefacto final S=3. `work/final.log` y `verificacion3/run/run.log` corresponden a S=3. `informe.tex:407-413` revela expresamente ambos titulares. Son antecedentes conservados, no prueba de falsificación; etiquetarlos evita usar el log viejo como reproducción final.

## Comprobación del protocolo y constructor nuevos, sin abrir puntuaciones

`PROTOCOL.md:19-28` corrige los riesgos principales: todos los pares elegibles, todas las respuestas de evaluación, ocho particiones, repetición aleatoria, grupos por contenido compartido, ajuste anidado, bootstrap por componente y multiplicidad. Reconoce que los datos públicos ya se inspeccionaron y que no es una prueba inédita. El diseño es apropiado para una revalidación local; no establece equivalencia con la plataforma privada.

La identidad `p²−2pq+q` en `build_independent.py:3-5` es exacta para predictores constantes por par y conserva la ponderación por respuesta dentro del par. `raw_aggregate` agrega por sujeto+ítem; `summarize_pair` ordena ítems y comprueba >=80. La simulación Bernoulli con `means[acquired]` equivale marginalmente a escoger uniformemente una de las respuestas registradas, para una sola respuesta por ítem. No reconstruye la identidad de la repetición concreta, que esos modelos no necesitan. La verificación de API del modelo original deberá hacerse separadamente: este constructor no lo ejecuta.

Detecté un fallo inicial de reanudación: meta con SHA válido y caché parquet ausente podía excluir silenciosamente un benchmark. Lo comuniqué; el agente principal añadió a la condición la existencia del parquet para soporte binario (`build_independent.py:115-117`, revisión posterior). PASA por inspección de la condición actual; no ejecuté un test de reanudación porque no poseo esos archivos.

Riesgos a comprobar por el agente principal: los cachés deben invalidarse si cambia la semántica de agregación, no solo si cambia SHA de respuestas; comprobar la integridad de metadatos e ítems además de respuestas; `pd.isna(value)` presupone metadatos escalares en `140`; componentes por hash exacto no detectan paráfrasis/casi duplicados; folds equilibrados en número de grupos no necesariamente en número de pares. Ninguno demuestra por sí mismo que el resultado nuevo esté mal. No se han leído esos resultados.

## Evidencia ejecutada y criterios de aceptación de esta revisión

Todos los comandos se ejecutaron desde PowerShell con `python -` y bloques propios; no se usaron scripts evaluadores originales.

| Criterio | Estado | Comando y salida resumida |
|---|---|---|
| Inventario de 15 scripts y14 resultados | PASA | `Get-ChildItem .../src -Name`; Python `len(list(src.glob('*.py')))` y `len(list(resultados.glob('*')))` ->15 y14. |
| Copias de resultados idénticas a work | PASA | Python `p.read_bytes()==(work/p.name).read_bytes()` para14 ficheros ->14 True. |
| ZIP conserva modelo y solo contiene model.py | PASA | Python `zipfile.ZipFile(...).namelist()` ->`['model.py']`; tamaño1596; bytes iguales a repo/submission/model.py. SHA256 del modelo `bf59ab3832e202cd217f5798bde62aa9be56d036f9cfcbbd561bcae75202434f`. No es ejecución del validador oficial. |
| ALC de resultados originales coherente con presupuestos | PASA | Python `max(abs(a.brier_alc-a[['B0','B1','B3','B7','B15','B31']].to_numpy() @ [.1,.2,.2,.2,.2,.1]))` ->5,55e−17. |
| Diferencias y86 victorias originales verificables | PASA | `pd.read_parquet` de res_beta_binomial_p005 y res_media_empirica_OFICIAL; merge1:1 sujeto/benchmark, groupby benchmark ->496 pares,101 benchmarks,86 diferencias<0; media por par−0,03488753; media benchmark−0,03470193; binomial exacta2,8269e−13. |
| Incertidumbre independiente sobre el orden original | PASA | Bootstrap propio: rng47,10.000 muestras de101 benchmarks con reemplazo, mantener sus pares y recalcular media ->IC95[−0,04106554,−0,02854009]. Alcance: artefactos guardados. |
| Cabecera aislada de ajustes de priors ricos | FALLA | `rg -n 'pred.ajusta|benchmark_id.*!= b' .../src/ejecuta.py` ->línea53 excluye solo b de agg completo. |
| Equivalencia >=80 respuestas y >=80 ítems | FALLA | Python propio `items=np.repeat(['i0','i1'],40)` ->len80, únicos2, primer criterioTrue, segundoFalse. Uso incorrecto localizado en sensibilidad:101 y predictores:201/239. |
| Identidad entre población ponderada y seis sujetos por benchmark | FALLA | Python `np.mean([-.01,.1])` ->+.045; `np.average([-.01,.1],weights=[1000,1])` ->−.0098901. La sensibilidad real requiere población completa. |
| “10 tests” prueban el modelo entregado | FALLA | Python `ast.walk(ast.parse(test_metrica.read_text()))`, lista imports ->harness y predictores, ningún submission/model.py. |
| Mejores cifras guardadas ocultadas | NO COMPROBABLE en general; evidencia local contraria | `Get-Content resultados_cabecera.csv`; `rg -n '0.2099|0.2102' informe.tex README.md` ->EB sujeto y macro publicados. No acceso a experimentos ausentes/borrados ni intención. |
| Ausencia general de señal de texto | NO COMPROBABLE | Límites ejecutables del experimento anterior: hashing sinIDF,alpha único,λ positivos y endpoint de oráculo retrospectivo. Sus resultados solo acotan las configuraciones ensayadas. |
| Igualdad con evaluación privada/plataforma | NO COMPROBABLE | No hay datos privados ni recibo de puntuación oficial dentro del alcance de esta revisión. |

## Riesgos fuera de los criterios anteriores

Las cifras de esta revisión confirman consistencia de resultados ya guardados, no honestidad histórica, integridad de datasets completos o rendimiento futuro. Los metadatos y respuestas binarios por soporte observado pueden diferir del contrato oficial de elegibilidad; deben documentarse sin reinterpretar etiquetas. El hash exacto no elimina toda relación entre benchmarks. La misma identidad de sujeto puede cambiar por fechas/harness; las conclusiones de transferencia por identidad no deben generalizarse a modelos de atributos. El Brier de prueba oficial, elegibilidad de participación y requisitos de publicación/subida necesitan evidencia externa separada. No se enviaron mensajes, publicaron archivos ni rotaron credenciales.

## Adenda: revisión del evaluador nuevo antes de mirar puntuaciones

Se revisó `evaluate_independent.py`, SHA256 `62ec3104fb917c865b920073fd0f91e2afc77ea0548a2a6905faa7af671b2df0`. No se ejecutó el evaluador ni se abrieron sus puntuaciones.

| Criterio | Estado | Evidencia concreta |
|---|---|---|
| Código sintácticamente válido | PASA | Python propio `ast.parse(path.read_text(encoding='utf8'))`, sin importar el módulo ->sin error. |
| Aislamiento exterior/interior en el código | PASA por inspección, ejecución pendiente | Líneas110-114 separan componentes completos;122/132/151 seleccionan por validación;129 ajusta tablas solo fit;140 ajusta vocabulario solo fit;144 ajusta Ridge solo fit;126/135/155-156 aplican al exterior. Los datos exteriores solo se usan como objetivo al puntuar. |
| Brier por respuesta y ALC | PASA para fórmula revisada y contrajemplo propio |19-28. Cálculo independiente con y=[0,1,1], p=[0,5;0,1;0,2;0,4;0,7;0,9]: error máximo entre expresión cerrada y MSE explícito1,6653e−16; ALC0,3526666667 igual a integración trapezoidal por índice de presupuesto. |
| Semillas no tratadas como8 unidades inferenciales | PASA por inspección |175 promedia primero por pareja;181 agrega sum/count por componente;183 reconstruye razón en cada bootstrap. Las8 filas de cada pareja pesan igual y se colapsan antes de inferencia. |
| Promoción considera multiplicidad | PASA por inspección |184 usa extremos0,05/(2×3) y1−0,05/(2×3);200 exige límite superior negativo y≥6/8 semillas favorables. No es evidencia de que alguna mejora haya pasado. |
| Ejecución completa, asserts de datos y artefactos nuevos | NO COMPROBABLE en esta revisión | Responsabilidad del agente principal; esta revisión deliberadamente no lee scores ni ejecuta tarea pesada concurrente. |

No detecté un defecto bloqueante en esa lógica de separación, métrica e inferencia. Quedan cuatro matices científicos:

1. Las tablas y Ridge no se reajustan con fit+validación tras seleccionar hiperparámetros: es válido y conservador respecto a tamaño de entrenamiento, aunque menos eficiente que un refit anidado. Mantener ese procedimiento congelado durante esta prueba evita adaptar decisiones a resultados externos.
2. La regularización de16 en tablas (`30-36`) se aplica a filas de escenarios, ocho por pareja. Equivale aproximadamente al peso de dos parejas en cada estado si las ocho caen en él, no a16 sujetos independientes. Esto afecta la fuerza efectiva de regularización, no convierte las semillas en unidades inferenciales porque después se promedian correctamente.
3. El bootstrap remuestrea pérdidas exteriores de modelos ya ajustados. Sus intervalos son condicionales a esos modelos, folds y selección; no incluyen la variabilidad de volver a entrenar/seleccionar toda la tubería en cada muestra. Los folds comparten entrenamiento. No presentar sus coberturas como garantía exacta para cualquier muestra futura.
4. El sign-flip por componente de186-188 sí usa magnitudes, con corrección+1 que evita p=0. Su interpretación inferencial requiere simetría/intercambiabilidad bajo la nula; no es una prueba aleatoria exacta universal para datos observacionales. Los intervalos por bootstrap y las sensibilidades son la evidencia principal.

## Adenda D: contexto de etiquetas adquiridas de otros sujetos

Revisión adicional solicitada expresamente por el agente principal. Leídos `EXPERIMENT_D.md`, `evaluate_context.py` y `context_model.py`; no se han abierto `context_summary.json`, `context_scores.parquet`, logs ni candidatos puntuados. Se autorizó además un único archivo nuevo, `revision_context_tests.py`, que conserva los tests sintéticos usados aquí. No se ha cambiado implementación ajena.

### Legitimidad y límites de la evidencia

El agente principal aportó el extracto oficial de https://aims.stanford.edu/competition/, apartado Test-time Adaptation: “labeled contains [[subject, item], label] entries across sampled subject–benchmark pairs.” El presupuesto0 tiene evidencia vacía. El repositorio oficial local es consistente con esa interfaz: `baseline/tools/streaming_ingestion.py:147` toma todo `event['labeled']` y `181` lo entrega a las llamadas de evaluación; `baseline/empirical_mean/model.py:26-34` filtra de esa evidencia por igualdad completa del sujeto y benchmark. No hay prohibición visible de usar las etiquetas ajenas ya entregadas. El código contextual usa exclusivamente ese argumento actual, sin recuperar etiquetas por archivos, estado entre llamadas o ID oculto.

Eso **permite** utilizar peers cuando aparecen; **no garantiza** que el coordinador entregue cuatro sujetos del mismo benchmark ni que todos tengan simultáneamente n etiquetas al checkpoint n. El coordinador de plataforma no es visible en el alcance revisado. La simulación de tamaños1/2/4/8 representa cuatro entornos condicionales de disponibilidad, no una estimación verificada del tamaño real en Codabench. La disponibilidad debe seguir marcada NO COMPROBABLE hasta una evaluación real o confirmación oficial. Elegir el tamaño que mejor puntúe después de observarlos introduciría selección del escenario; la cohorte4 ya está predefinida como principal.

`evaluate_context.py:12-26` suma únicamente los éxitos adquiridos k de los miembros de una cohorte del mismo benchmark y semilla. No usa `q_eval` para construir contexto; este solo entra al puntuar y seleccionar hiperparámetros interiores. Cada sujeto recibe n etiquetas por presupuesto; el total de una cohorte de tamaño c es c×n, sin comprar etiquetas adicionales fuera de esos presupuestos simulados. Las colas de cohortes parciales usan su tamaño real. Las particiones de ítems siguen siendo las del constructor original de la auditoría; se mantiene pendiente confirmar que la plataforma usa exactamente esa organización de pools.

### Identidad visible y referencia correcta

Conteo propio sobre `pairs.parquet`, sin mirar resultados: **8 grupos de atributos visibles idénticos dentro de un mismo benchmark,59 pares**; tamaños [10,19,3,3,2,3,9,10]. Afectan a ceval,devbench,elicitation_game,emoji_attack yvl_rewardbench. Globalmente hay241 firmas repetidas que abarcan632 pares, pero la comparación por benchmark impide confundir sujetos de benchmarks distintos.

La firma incluye los ocho atributos de la API, sin IDs ocultos. No hay valores nulos en esas ocho columnas de `pairs.parquet`. El contrato oficial aportado por el agente principal normaliza valores ausentes como cadena vacía y todos los atributos como strings, coherente con `fillna('').astype(str)` del constructor/simulador. Si la interfaz real entregara None, claves ausentes, tipos distintos o campos extra, la igualdad de diccionarios de `context_model.py:20` podría diferir de esa firma; es una dependencia explícita del contrato, no un problema observado en los datos preparados.

La agregación de sujetos con atributos indistinguibles es la que haría la API original al comparar diccionarios. Por eso `original_cohort2/4/8` es una referencia más fiel a esa disponibilidad que el `original_beta3` del análisis previo, que operaba por ID oculto. En esos59 pares ambas referencias pueden diferir. Cada mejora D debe compararse contra `original_cohortN` del mismo escenario; mezclarla con la referencia inicial confundiría mejora contextual, disponibilidad y corrección de identidad. Cohorte1 coincide con el caso sin otras etiquetas disponibles.

### Selección interior, fallback y pruebas ejecutadas

No detecté una fuga de prueba en el código de D. Los estados `(k,n)` de contexto son deterministas a partir de evidencia adquirida. La rejilla rho0/0,25/0,5/1 yS1/3/8/16 se elige solo con benchmarks de validación interior de `fold-*-roles.json`; esos roles proceden del protocolo previamente congelado y se verificó que excluyen el exterior. No hay ajuste con objetivos externos. La misma configuración elegida bajo cohorte4 se aplica a los otros tamaños, sin elegir de nuevo por sus puntuaciones externas.

La familia libre `context` puede cambiar S y **no garantiza** fallback al modelo original cuando no hay peers. La familia `fallback_safe` fijaS3 y solo elige rho; cuando peer_n=peer_k=0, m_peer=0,5 y la predicción coincide exactamente con `(1,5+k)/(3+n)`, incluso si hay varias filas indistinguibles agregadas como evidencia propia. El código exportable actual contieneS3 yPEER_MIX0,5 como valor de investigación: aún debe elegirse/documentarse el rho final por el procedimiento interior y verificar el artefacto exportado. Un modelo final único no puede copiar cinco hiperparámetros distintos de los folds sin una regla final de ajuste predefinida.

Comando ejecutado: `python E:\AIMS-sprint\work\auditoria_codex_20261024\revision_context_tests.py`.

| Criterio | Estado | Salida/evidencia concreta |
|---|---|---|
| Paridad simulador de cohortes/API real contextual | PASA |5376 comparaciones, error máximo0. Incluyen todosrho/S de la rejilla, presupuestos0/1/3/7/15/31, cohortes1/2/4/8, siete sujetos sintéticos, atributos duplicados y colas de3/7. Se construyen explícitamente las etiquetas y se llama al predictor, no al evaluador completo. |
| Fallback sin peers para todos los estadosn0..31,k0..n | PASA |528 casos de igualdad exacta con Beta(1,5;1,5). |
| Roles interiores/exteriores | PASA |5 manifiestos: conjuntosfit/validation/test disjuntos; test corresponde alfold exterior; validación lo excluye. |
| Conteo8grupos/59pares indistinguibles | PASA |Conteo independiente por benchmark+JSON de los ocho atributos:8grupos,59pares,[10,19,3,3,2,3,9,10]. |
| Evidencia de evaluación usada como contexto | PASA por inspección: no detectada |`cohort_evidence` solo recibe pairs,d,k yusa k de adquisición. q_eval solo entra en alc/validación. |
| Disponibilidad real de cohorte4 | NO COMPROBABLE |La interfaz permite evidencia de pares diversos pero no fija esa cantidad/simultaneidad. No se consultó Codabench ni se abrió su recibo privado. |
| Ajuste de multiplicidad para elegir entre todas las familias | FALLA inicial; PASA por inspección tras edición del principal |La revisión original calculabaIC98,75% para4familias aunque conservaba context yfallback_safe además de3familias anteriores. Si ambos pueden ser promovidos por resultados externos, son5 candidatos. Último `rg -n 'ci99|\.05/10' evaluate_context.py` confirma en85 `ci99_five_candidates` con extremos0,05/10 y1−0,05/10. No se han abierto resultados. |

La matriz previa de métricas no sustituye esta prueba de disponibilidad. Los IC por componente siguen siendo condicionales a folds/modelos ya ajustados y a la cohorte simulada. Si el ajuste se amplía a5candidatos, debe abarcar también los3contrastes previos que intervengan en una selección conjunta. Los tamaños1/2/8 no requieren presentarse como descubrimientos separados si permanecen sensibilidades predefinidas; si se escoge libremente un tamaño o se anuncian varias mejoras por tamaño, haría falta considerar esa selección adicional.

Huellas de las versiones efectivamente probadas:

- EXPERIMENT_D.md: `2a523a8db3ae1c2c9b836aee7db3ff35d5aad2a79089290d259f73a16b8638cf`.
- evaluate_context.py: `71014bd65891656abb25a6e8efac95aa00d5db827bd75b9f22a611dfa0d6dffc`.
- context_model.py: `64d53ec47d30242f009cdbae387df3b96c64237c291246787b70730ee4a85803`.

No detecté errores bloqueantes de implementación en la paridad probada. La multiplicidad inicial se corrigió en el código por el agente principal: se inspeccionó la fórmula deIC99% yla canonicalización de fallback de69, ejecutada solo después de probar igualdad exacta de probabilidades y tolerancia1e−14 al puntuar. Esto corrige diferencias de redondeo entre lotes, no convierte un resultado distinto en igualdad. Se mantiene pendiente verificar el resultado de la tubería completa y el ajuste conjunto en los otros contrastes. La disponibilidad real sigue limitando una eventual promoción. No se ha visto ninguna puntuación D para emitir este juicio.
