# Experimento adicional D: etiquetas de otros sujetos

Registrado después de ver los resultados de las tres familias iniciales y antes de calcular ninguna puntuación de este experimento. Es un experimento exploratorio motivado por una capacidad explícita de la interfaz oficial: `labeled` contiene observaciones de distintos pares sujeto–benchmark. No se presupone que el servidor entregue siempre todos los pares simultáneamente.

| Factor | Diseño fijado |
|---|---|
| Cohorte principal | 4 sujetos por benchmark; bloques aleatorios deterministas por semilla |
| Sensibilidad | 1,2,8 sujetos; tamaños reales menores al final del bloque |
| Datos | Mismos Parquet, 8 semillas y folds/componentes bloqueados del protocolo principal |
| Evidencia propia | Agregar etiquetas de atributos visibles exactamente iguales; auditar colisiones de identidad |
| Evidencia ajena | Solo otros atributos visibles y mismo benchmark anónimo; nunca etiquetas de evaluación |
| Prior contextual | m_peer=(1,5+k_otros)/(3+n_otros); m=0,5+rho*(m_peer-0,5) |
| Predicción | (k_propio+S*m)/(n_propio+S) |
| Rejilla | rho=0,0,25,0,5,1; S=1,3,8,16 |
| Elección | Validación interior por componente; una misma configuración elegida con cohorte4 se prueba en 1,2,4,8 |
| Control de fallback | Una variante de S=3 con rho elegido, que coincide exactamente con el original si no hay evidencia ajena |
| Inferencia | 10.000 bootstrap por componente; IC95 y IC98,75 para cuatro familias; escenarios completos separados |
| Promoción | Además de incertidumbre favorable, no empeorar en ausencia de evidencia ajena; declarar disponibilidad real no verificada en Codabench |

Las ocho configuraciones de sujeto con atributos visibles coincidentes, que afectan a 59 pares, se tratan por igualdad de atributos; no se distingue un ID oculto que el predictor no recibe. La prueba de igualdad con la API del modelo original y de la variante contextual se realizará mediante evidencia sintética, incluidos sujetos con atributos iguales.

No se usarán puntuaciones de este experimento para seleccionar de nuevo las tres familias previas. Se conservarán todas las configuraciones interiores y resultados, y se etiquetará el carácter exploratorio en el informe.
