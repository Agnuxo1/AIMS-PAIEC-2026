# CIERRE_AIMS — estado verificado de cada entregable

**Estado global: PENDIENTE — BLOQUEADO EXTERNAMENTE.**
El concurso **no** está entregado ni cerrado. Falta el ID de envío de Codabench y, por tanto, el envío a OpenReview. El código, los resultados, el informe y el vídeo están publicados. El perfil de OpenReview ya está creado y pendiente de activación manual.

Fecha de este documento: **2026-10-24T09:35:59+00:00**.
Plazo oficial: **30-10-2026 23:59 AoE = 31-10-2026 11:59 UTC** (≈ 7,1 días desde esta fecha). Verificado hoy en la fuente oficial; no ha cambiado.

---

## 1. Repositorio público — PUBLICADO

| Dato | Valor |
|---|---|
| URL | <https://github.com/Agnuxo1/AIMS-PAIEC-2026> |
| Visibilidad | **Público** (comprobado sin sesión en un navegador aislado) |
| Cuenta | `Agnuxo1`, la verificada. No había repositorio previo de esta participación |
| Commit del cierre | `16bc039` (base verificada antes de esta actualización; ver Git para el commit que la contiene) |
| Licencia | MIT (`LICENSE`), terceros en `THIRD_PARTY.md` |
| Contenido | Base inicial de 123 archivos, más la evidencia de cierre: entrega, auditoría independiente, protocolo, resultados con candidatos desfavorables, informe PDF y LaTeX, descargador fijado, los 34 originales en `legacy_original/` |
| Fuera de Git | datos brutos, cachés, entornos y credenciales |

**Release:** <https://github.com/Agnuxo1/AIMS-PAIEC-2026/releases/tag/v1.0.0>

| Artefacto | Bytes | SHA-256 |
|---|---:|---|
| `submission_original.zip` | 1.596 | `895c3a9f693947ab096fb3c75120858416594b95c69261b5fc67207fa79722f8` |
| `submission_context_experimental.zip` | 736 | `eb52cd786875533e0f506723ed7d5ed06ec0d7d09f05c96dd8512a3ad630a94b` |
| `technical_report_v2.pdf` | 177.611 | `3fc899c8bfe66564711ca29f4a4cbe207df462462b5c62d76ed0cecf2da5a04f` |

El informe de la tabla anterior pertenece a la release inicial, conservada intacta. El [PDF actual](report/informe_v2.pdf) incorpora la URL pública del repositorio: **181781 bytes**, SHA-256 `5510cd7318090db52ae8ec1bf92fd9b8561171cc8b7dad1f18a2ea87ab9aaea1`, cinco páginas. No cambia los resultados científicos.

## 2. Integración continua — EN VERDE

| Dato | Valor |
|---|---|
| Último run | [40082766428](https://github.com/Agnuxo1/AIMS-PAIEC-2026/actions/runs/40082766428) · **success** |
| Qué ejecuta | pruebas sintéticas de métrica e inferencia; validación de los dos ZIP; regeneración de figuras y GIF |
| Qué NO hace | no descarga los 5,35 GB, no usa credenciales ni servicios de pago |

El primer run **falló** (`37690ab`): `scikit-learn==1.4.0` fija `numpy<2` y el fichero de requisitos era insatisfacible en limpio, aunque ambos convivían en la máquina local. Corregido en `e75745c` fijando la versión compatible y registrando las versiones reales en `docs/environment_used.json`. Se deja constancia porque el run fallido sigue en el historial.

## 3. Gráficas y GIF — PUBLICADOS

Todo se genera desde `results/*.json` y `results/*.csv`; ninguna cifra está escrita a mano en los generadores.

| Archivo | Qué muestra |
|---|---|
| `docs/figures/fig1_budget_curve.png` | Brier por presupuesto; dónde se rompe la referencia oficial |
| `docs/figures/fig2_methods.png` | Cada método con su intervalo del 95 % |
| `docs/figures/fig3_cohorts.png` | Ganancia condicional según cuántos sujetos son visibles |
| `docs/figures/fig4_status.png` | Los tres tipos de afirmación, separados |
| `docs/figures/gif1_shrinkage.gif` | Cómo mueve la predicción cada etiqueta revelada (32 fotogramas) |
| `docs/figures/gif2_peers.gif` | Cómo desplaza el prior la evidencia de otros sujetos (4 fotogramas) |
| `docs/figures/gif1_shrinkage_still.png` | Imagen estática alternativa |

Los dos GIF están marcados en su propio fotograma como ilustración sintética. Generadores en `build/`. Comprobado que se reproducen en la página pública de GitHub.

## 4. Vídeo — PUBLICADO Y CORREGIDO

| Dato | Valor |
|---|---|
| URL | <https://youtu.be/dBfUTN5afcQ> |
| Canal | el verificado de Fran (`UCpmNBgiKW_NxkF1BBZ6v6NQ`) |
| **Visibilidad** | **Público**, observado en la configuración guardada de Studio el 24-10. El cierre anterior lo describía como no listado |
| Duración | 4:38 · 1920×1080 · H.264 · AAC 48 kHz estéreo · 9,0 MB |
| Narración | Inglés, voz neural `en-GB-RyanNeural` vía `edge-tts` (gratuito, sin servicio de pago) |
| Miniatura | Personalizada, aplicada y comprobada |
| Audiencia | Marcado como no creado para niños |
| Materiales | Guion, narración, scripts de render, miniatura y tiempos en `docs/video/` |

**Subtítulos.** El MP4 **no tiene ninguna pista de subtítulos** (comprobado con `ffmpeg -i`: 0 pistas) y no hay transcripción incrustada ni rótulos con la narración. Solo hay títulos breves y etiquetas de gráficas, que el encargo permite.

**Corrección verificada por Codex.** Se guardaron el idioma original y el del título/descripción como **Inglés**. Se eliminó la pista automática desde Idiomas → Detalles de Inglés → Subtítulos automáticos → Eliminar → Actualizar. Al reabrir el diálogo guardado ya no aparece esa pista. El reproductor de la página del vídeo indica **«Subtítulos no disponibles»**. Evidencia: [captura de Studio](docs/evidence/youtube_sin_subtitulos.png).

## 5. Registro y envío al concurso — BLOQUEADO EXTERNAMENTE

| Hito | Estado |
|---|---|
| Cuenta de Codabench | **No confirmada.** Un correo del 7-10-2025 demuestra que Fran **tiene** cuenta en Codabench, pero no revela el nombre de usuario y la plataforma no es accesible para consultarlo. **No lo invento** |
| Registro del equipo | **Preparado, sin enviar.** Datos conocidos rellenados; falta confirmar el usuario de Codabench |
| Envío del ZIP | **No hecho** |
| **Submission ID** | **NINGUNO** |
| Puntuación oficial | **NINGUNA** |
| Perfil de OpenReview | **CREADO**: `~Francisco_Angulo_de_Lafuente1`. Registro confirmado a las 12:15:07 UTC; correo profesional confirmado y actualización del perfil recibida a las 12:24:03 UTC. **Activación pendiente de moderación** |
| Informe en OpenReview | **No enviado.** Faltan la activación del perfil y un submission ID aceptado de Codabench |

**Evidencia del bloqueo.** `www.codabench.org` resuelve por DNS a `129.175.8.21` (Paris-Saclay) pero el puerto 443 agota el tiempo. Comprobado **doce veces** entre el 23-10 21:30 UTC y el 24-10 09:09 UTC, desde esta red, desde un verificador HTTP externo independiente y con resolución DNS por 1.1.1.1 y 8.8.8.8. En las mismas máquinas y momentos, `aims.stanford.edu`, Hugging Face y GitHub responden con normalidad.

**Comunicación con los organizadores.** Correo enviado el **24-10-2026 a las 11:12 (Madrid)** a `aimslab@cs.stanford.edu`, asunto *«Codabench unreachable - Predictive AI Evaluation Competition submission route»*, desde `lareliquia.angulo@gmail.com`. Verificado en la carpeta Enviados. Antes se comprobó que **no existía un hilo previo** con los organizadores, para no duplicar. Pregunta por: si la caída es conocida y su previsión, si hay una vía alternativa escrita que conserve la elegibilidad, cómo gestionar el campo del submission ID, y qué conviene preparar. **Enviar ese correo no equivale a una aceptación**: sin respuesta explícita de los organizadores que habilite otra vía, el estado sigue siendo bloqueado.

**Depósito del material final por correo.** El 24-10-2026 a las **09:40:08 UTC (11:40 Madrid)** se respondió en el mismo hilo, adjuntando el PDF actualizado (181.781 bytes) y los dos ZIP (1.596 y 736 bytes), con sus SHA-256 y el enlace al código fijado en `16bc039`. Se identificó `submission_original.zip` como candidatura principal y el contextual como experimental. Los tres adjuntos constan en Enviados. Se solicitó confirmación de recepción y de la posible vía alternativa. **Este depósito está enviado; su aceptación como entrega oficial no está confirmada.**

**Registro de OpenReview completado.** Se incorporaron el historial profesional real y sus fuentes públicas, ORCID, una página de autor con el correo del registro, y la actividad actual en P2PCLAW. `research@p2pclaw.com` se confirmó mediante el código enviado por OpenReview, conservando Gmail como correo preferido. El segundo aviso de OpenReview confirma que el mismo perfil sigue en la cola de moderación. Esto acredita el alta del perfil, **no** la aceptación del informe ni una participación puntuada. No se guardan códigos de verificación, enlaces de activación ni fecha de nacimiento en el repositorio.

## 6. Resultado que se reclama, y el que no

| Afirmación | Estado |
|---|---|
| 0,236814 → 0,205984 Brier ALC, 133/151 benchmarks, IC95 [−0,038426, −0,027214] | **Validación local** en benchmarks públicos retenidos entre sí, con el protocolo independiente |
| Mejora condicional adicional de −0,002562 (IC99 [−0,006257, −0,000551]) | **Experimental**, condicionada a que la plataforma suministre etiquetas de otros sujetos, lo cual **no está verificado**. Estima el procedimiento de selección anidada, no es una prueba inédita del parámetro final |
| Puntuación oficial, ranking o aceptación | **No se reclama ninguna**, en ningún archivo del repositorio, del vídeo ni de la release |

## 7. Obligaciones futuras anunciadas por el concurso

- Sesión del concurso: **viernes 11-12-2026, Sídney, presencial, 08:00–10:15** hora local. Cinco orales que suman 60 min, más una sesión de pósteres de 60 min. No consta apoyo a viaje.
- Para optar a oral o póster hace falta informe técnico **y** publicar el código usado. El código ya está publicado.
- Los organizadores reejecutan el método liberado para comprobar reproducibilidad.

## 8. Siguiente paso exacto

1. Reintentar `https://www.codabench.org/competitions/17828/` de forma espaciada hasta el 30-10.
2. Al recuperar acceso: confirmar el usuario de Codabench, completar el formulario de registro del equipo, subir `submission_original.zip`, anotar el submission ID y su estado (carga ≠ ejecución terminada ≠ aceptación válida).
3. Si la cuota lo permite y las reglas lo admiten, probar también el archivo contextual y comprobar si la plataforma suministra de verdad etiquetas de otros sujetos — **sin** intentar identificar benchmarks o sujetos anónimos.
4. Seleccionar la entrega final con la evidencia disponible y documentar la decisión.
5. Enviar por OpenReview el informe con el submission ID elegido y los enlaces públicos.
6. Si llega respuesta de los organizadores con una vía alternativa, seguirla y registrar el recibo.

## 9. Advertencia de seguridad pendiente

El archivo `UTILIDADES_HERRAMIENTAS_APIs.txt` quedó expuesto en una sesión anterior. No se ha leído ni republicado en esta tarea. **Ninguna clave, cookie, registro de autenticación ni historial privado aparece en el repositorio, el vídeo, la release ni este documento.** La rotación de claves sigue pendiente de decisión del titular.

---

*Documento factual. Cada URL y cada estado de este archivo se comprobó en la sesión del 24-10-2026. Donde algo no está hecho, dice que no lo está.*
