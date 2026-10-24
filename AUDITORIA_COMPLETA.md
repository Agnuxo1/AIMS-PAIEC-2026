# Auditoría independiente AIMS y mejoras — 24 de octubre de 2026

## Resultado comprobado

La mejora original es real dentro de su diseño local: **0,2453845123 → 0,2104969827**. He reconstruido sus 769 pares desde los datos y reproducido los 496 pares de cabecera; el error máximo por par es 1,11×10⁻¹⁶. No se ejecutó su evaluador ni su pickle para obtener esa reproducción.

**No hay evidencia observable de ocultación de mejores resultados guardados.** El Empirical Bayes por sujeto (0,20989535) y el jerárquico macro (0,21024007) ya aparecen en la tabla del informe; el primero también encabeza el README. No se puede recuperar lo borrado o no guardado ni deducir intención del autor.

Sí había problemas que corregir. Ahora hay una evaluación independiente y una candidata mejorada, con un resultado favorable **condicionado a disponer de etiquetas de otros sujetos**.

## Qué se ha auditado

- Los 15 scripts originales, 14 resultados, modelo, informe, README y logs de trabajo/verificaciones. Una segunda revisión examinó código y metodología sin ver las nuevas puntuaciones.
- Las respuestas públicas desde Parquet, con agregación independiente por sujeto/ítem, elegibilidad real y 8 semillas. No se importaron los módulos de análisis anteriores ni se abrió `pares.pkl`.
- **890 archivos públicos fijados por SHA-256**, contrastados con sus registros de descarga. Revisión real `54074453da7b9911be0f48e4c560be547bc84d21`; el manifiesto descarga exactamente 5.354.371.887 bytes de entradas necesarias.
- La fuente del baseline local coincide con la oficial tras normalizar saltos de línea; su árbol sintáctico es idéntico.

El commit citado antes no describe completamente la carpeta descargada. Sin embargo, los aciertos y recuentos de los 14.733 pares anteriores coinciden exactamente con la nueva reconstrucción. Solo se añaden dos pares constantes, ineligibles, con 116 respuestas. Hay 70.193.656 filas físicas en tablas binarias y 68.840.604 respuestas no ausentes: no son el mismo recuento.

## Defectos identificados y tratamiento

| Hallazgo | Consecuencia y corrección |
|---|---|
| LOBO de modelos ricos excluía solo el benchmark objetivo, usando otros benchmarks de cabecera al ajustar durante desarrollo | No era una separación desarrollo/cabecera plenamente aislada. Nuevo ajuste y selección anidados por componentes. El Beta fijo no aprende de esos agregados y no hereda directamente esa fuga. |
| Algunos filtros usaban 80 respuestas en vez de 80 ítems distintos | Hay 90 pares públicos en esa situación; en la cabecera anterior el conteo correcto es 11.639, frente a 11.657 por respuestas. Nuevo criterio exacto. |
| Seis sujetos y 150 ítems de evaluación por benchmark/par | Cambia población y ponderación. Nueva evaluación con todos los 13.200 pares elegibles y todas sus respuestas de evaluación. |
| Conclusiones generales a partir de negativos de texto muy específicos | Hashing sin IDF, muestreo por respuesta y endpoint diferente. Se conservan como fallos de esas configuraciones; no prueban ausencia de señal transferible ni necesidad de APIs de pago. |
| Intervalo de otra trayectoria y nombre impreciso de prueba | No mezclar el IC de adquisición aleatoria con el titular. La prueba anterior era binomial de signos. Bootstrap nuevo por componente, con semillas promediadas antes del remuestreo. |
| Oráculo empírico presentado como suelo de ruido | Un ajuste retrospectivo a las respuestas observadas no demuestra ruido irreducible en respuestas nuevas. Redacción corregida. |
| Descargador sin revisión fijada, rutas absolutas, licencia MIT declarada sin LICENSE | Nuevo descargador con revisión y hashes, variable AIMS_DATA, requisitos registrados, LICENSE y documentación de terceros. |
| IDs internos diferentes con atributos visibles iguales | 8 grupos, 59 pares. La simulación contextual y la API comparan diccionarios visibles, tal como exige la interfaz. |

## Nuevas pruebas y mejoras

El protocolo principal se guardó antes de calcular nuevas puntuaciones. Incluye **151 benchmarks, 13.200 pares, 105.600 escenarios, 142 componentes de contenido, cinco folds externos y validación interior**. Los benchmarks conectados por hashes exactos no se separan. La métrica principal promedia pares, no benchmarks ni respuestas.

La reconstrucción completa vuelve a confirmar el baseline Beta: **0,236814 → 0,205984**, diferencia −0,030830 e IC95 [−0,038426, −0,027214]; mejora en 133 de 151 benchmarks. Estas cifras corresponden a otro protocolo y no sustituyen numéricamente el 14,2 % original sin explicar el cambio.

Primeras tres familias, con evidencia aislada por par y referencia Beta 0,205984:

| Variante | Brier ALC | Diferencia | IC95 por componente |
|---|---:|---:|---|
| Prior Beta ajustado | 0.205535 | -0.000449 | [-0.005350, +0.003663] |
| Tabla de calibración | 0.205877 | -0.000107 | [-0.000522, +0.000105] |
| Prior Ridge de atributos | 0.216549 | +0.010565 | [+0.001218, +0.017943] |

Las dos primeras ganancias puntuales no son concluyentes; la tercera empeora. Todos los hiperparámetros y sus puntuaciones de validación se guardan, incluidos resultados desfavorables.

### Extensión contextual registrada por separado

La interfaz permite recibir observaciones de distintos pares sujeto–benchmark. Registré después del experimento principal, pero antes de calcular sus resultados, una prueba que utiliza **exclusivamente etiquetas suministradas de otros sujetos del mismo benchmark**, nunca sus respuestas de evaluación.

Se probaron dos variantes más: permitir cambiar también la fuerza del prior, y mantener fuerza 3 para volver exactamente al original si no hay etiquetas ajenas. En cohortes simuladas de cuatro sujetos:

| Método | Brier ALC | Interpretación |
|---|---:|---|
| Original con la misma evidencia y atributos visibles | 0,206027 | Referencia correcta de este entorno |
| Contextual con fuerza reajustada | 0,207251 | Empeora; se conserva el resultado |
| **Contextual con fuerza 3 y retorno exacto** | **0,203465** | **Mejora adicional de 1.24 %** |

La última diferencia es **−0,002562**, IC95 [−0,005234, −0,000938] y **IC99 [−0,006257, −0,000551]** al ajustar por los cinco candidatos comparados. Es favorable en las ocho semillas. Con dos sujetos la ganancia puntual es −0,001298 y con ocho es −0,003465; son sensibilidades de disponibilidad, no una selección del entorno más favorable.

Con un solo sujeto las probabilidades son **exactamente iguales** al original. La candidata exportada usa fuerza 3 y mezcla 0,5, elegida con puntuaciones interiores registradas, sin utilizar las puntuaciones externas. El 0,203465 estima el procedimiento de selección anidado, no una prueba inédita posterior del parámetro final ajustado con desarrollo público.

**Límite decisivo:** todavía no sabemos cuántas etiquetas de otros sujetos entrega simultáneamente Codabench. La mejora es local y condicional, no puntuación oficial. Mantengo ambos ZIP. No he sustituido la entrega original ni enviado ninguno.

## Validación ejecutada

- 150 comprobaciones independientes de la identidad exacta de Brier frente a errores cuadráticos explícitos.
- Los 63 estados n,k de los seis presupuestos oficiales y sus invariantes contra el modelo entregado.
- **5.376 comparaciones simulador/API contextual, error máximo cero**, con cohortes 1/2/4/8, colisiones visibles y grupos parciales.
- **528 estados válidos n=0…31, k=0…n con retorno exacto** al original.
- Pruebas de agregación a través del límite de 250.000 filas, valores ausentes, rechazo de no binarios, invariancia al orden de almacenamiento y ponderación por pares.
- Cinco manifiestos de fitting/validación/prueba disjuntos, selección solo interior y todos los resultados conservados.
- Informe técnico nuevo de **cinco páginas**, plantilla oficial NeurIPS, compilación final sin avisos y revisión visual de todas las páginas.

Se conservan los fallos de instrumentación de esta auditoría: una primera réplica asumió erróneamente el orden de salida de una agrupación hash; se corrigió leyendo el orden de los datos originales. También se corrigieron una conversión JSON de entero NumPy y un chequeo bit a bit de puntuaciones que debía admitir redondeo BLAS. No se modificaron las rejillas para eliminar resultados negativos.

## Alcance e incertidumbre

El bootstrap condiciona a los modelos ya ajustados; no reajusta toda la selección en cada réplica. Los hashes exactos no detectan paráfrasis. Los folds están equilibrados por número de componentes, no por pares, y unos pocos benchmarks concentran buena parte de la población. La población pública puede diferir de los sujetos que muestree la plataforma. Los datos públicos ya se habían visto antes: no son una prueba prospectiva inédita. No se demuestra que no existan otras mejoras, ni que las alternativas de texto estén agotadas.

Las pruebas se ejecutaron en CPU, con bibliotecas limitadas a cuatro hilos y una tarea pesada cada vez. Servidor-ia tenía un trabajo ajeno activo. El usuario autorizó después la RTX 3090 local como alternativa; no fue necesario usar ninguna GPU ni pagar inferencia.

## Entrega y estado externo

El paquete v2 incluye código nuevo, protocolos, manifiestos, puntuaciones por par, todos los candidatos de validación, revisión independiente, informe PDF/fuente/estilo, requisitos, licencia y copia íntegra del repo original. El ZIP contextual solo contiene model.py; el ZIP original se conserva byte a byte.

Siguen pendientes las cuentas/registro, publicación del repositorio, enlace público en el informe, prueba aceptada y submission ID de Codabench, y envío por OpenReview. El correo anterior sigue sin enviar. Las credenciales expuestas no se han leído, probado ni rotado.

Fuentes oficiales: [reglas e interfaz](https://aims.stanford.edu/competition/), [baseline](https://github.com/aims-foundations/paiec_baseline), [datos](https://huggingface.co/datasets/aims-foundations/measurement-db).
