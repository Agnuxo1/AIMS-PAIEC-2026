"""Predictores para el PAIEC, en el orden en que los compara el informe.

1. `constante_mitad`   — predice 0.5 siempre. Brier = 0.25. Suelo trivial.
2. `media_empirica`    — EL BASELINE OFICIAL: 0.5 sin etiquetas, media cruda después.
                         Es deliberadamente ingenuo: con n=1 predice 0 o 1 y un
                         fallo cuesta 1.0 de Brier. Batirlo NO demuestra que el
                         método sea competitivo, y el informe lo dirá así.
3. `beta_binomial`     — CONTROL FUERTE Y BARATO: posterior Beta(a,b) con prior
                         informado y contracción automática con n pequeño.
4. `jerarquico`        — prior informado por atributos del sujeto y estadística
                         del benchmark + contracción beta-binomial.

Todos reciben (par, labeled_idx) y devuelven un array de probabilidades, una por
ítem DISTINTO del pool de evaluación del par.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-6


def _clip(p):
    return np.clip(p, EPS, 1.0 - EPS)


# --------------------------------------------------------------------- triviales

def constante_mitad(par, labeled_idx):
    return np.full(len(par.eva_items), 0.5)


def media_empirica(par, labeled_idx):
    """Baseline oficial: media cruda, sin suavizar ni recortar. 0.5 si no hay."""
    if len(labeled_idx) == 0:
        return np.full(len(par.eva_items), 0.5)
    y = par.adq_y[labeled_idx]
    return np.full(len(par.eva_items), float(y.sum()) / len(y))


# ----------------------------------------------------------------- beta-binomial

class BetaBinomial:
    """Posterior Beta con pseudo-observaciones: p = (a + k) / (a + b + n).

    `fuerza` = a + b es cuántas observaciones vale el prior. Con n=1 y fuerza 8,
    una etiqueta mueve la predicción poco; con n=31 la domina la evidencia.
    El prior `p0` puede ser global o venir de un modelo (ver `Jerarquico`).
    """

    def __init__(self, p0: float = 0.5, fuerza: float = 8.0):
        self.p0, self.fuerza = p0, fuerza

    def prior(self, par) -> float:
        return self.p0

    def __call__(self, par, labeled_idx):
        p0 = self.prior(par)
        a = p0 * self.fuerza
        b = (1.0 - p0) * self.fuerza
        n = len(labeled_idx)
        k = int(par.adq_y[labeled_idx].sum()) if n else 0
        p = (a + k) / (a + b + n)
        return np.full(len(par.eva_items), _clip(p))


# -------------------------------------------------------------------- jerárquico

class Jerarquico(BetaBinomial):
    """Prior por par estimado fuera del benchmark + contracción beta-binomial.

    El prior combina, en escala logit:
      - media global de aciertos (ajustada fuera del benchmark objetivo),
      - efecto del sujeto: su tasa media en los OTROS benchmarks,
      - efecto del benchmark: su dificultad media sobre los OTROS sujetos.
    Es un modelo de dos vías con contracción, no una IRT: los ítems del test
    pertenecen a benchmarks que no se han visto, así que no hay parámetros de
    ítem que transferir. Ver PLAN.md seccion 2.
    """

    def __init__(self, fuerza: float = 8.0, k_sujeto: float = 50.0, k_bench: float = 50.0):
        super().__init__(0.5, fuerza)
        self.k_sujeto, self.k_bench = k_sujeto, k_bench
        self.global_logit = 0.0
        self.ef_sujeto: dict[str, float] = {}
        self.ef_bench: dict[str, float] = {}

    @staticmethod
    def _logit(p):
        p = min(max(p, 1e-4), 1 - 1e-4)
        return float(np.log(p / (1 - p)))

    def ajusta(self, tabla):
        """`tabla`: DataFrame con subject_id, benchmark_id, k (aciertos), n (total).

        Debe venir YA SIN el benchmark que se va a evaluar (LOBO lo garantiza).
        """
        g = float(tabla["k"].sum()) / max(1.0, float(tabla["n"].sum()))
        self.global_logit = self._logit(g)

        s = tabla.groupby("subject_id")[["k", "n"]].sum()
        # contracción hacia la media global proporcional a cuántas observaciones hay
        ps = (s["k"] + self.k_sujeto * g) / (s["n"] + self.k_sujeto)
        self.ef_sujeto = {i: self._logit(v) - self.global_logit for i, v in ps.items()}

        b = tabla.groupby("benchmark_id")[["k", "n"]].sum()
        pb = (b["k"] + self.k_bench * g) / (b["n"] + self.k_bench)
        self.ef_bench = {i: self._logit(v) - self.global_logit for i, v in pb.items()}
        return self

    def prior(self, par) -> float:
        z = self.global_logit + self.ef_sujeto.get(par.subject_id, 0.0)
        # el benchmark objetivo NO está en ef_bench bajo LOBO: su efecto es 0,
        # que es exactamente la situación de arranque en frío del concurso.
        z += self.ef_bench.get(par.benchmark_id, 0.0)
        return float(1.0 / (1.0 + np.exp(-z)))


# ------------------------------------------------------------------- ablaciones

class SoloIdentidad(Jerarquico):
    """Ablación: usa SOLO el efecto del sujeto (identidad del sistema).

    Mide DEPENDENCIA DE VARIABLES, no demuestra fuga: el proveedor y la fecha
    pueden ser proxies legítimos de capacidad.
    """

    def prior(self, par) -> float:
        z = self.global_logit + self.ef_sujeto.get(par.subject_id, 0.0)
        return float(1.0 / (1.0 + np.exp(-z)))


class SoloGlobal(Jerarquico):
    """Ablación: sin sujeto ni benchmark, solo la media global."""

    def prior(self, par) -> float:
        return float(1.0 / (1.0 + np.exp(-self.global_logit)))


# ------------------------------------------------- jerárquico con macro-promedio

class JerarquicoMacro(Jerarquico):
    """Como `Jerarquico`, pero promediando igual que la métrica oficial.

    Hallazgo del sprint: el prior global ponderado por respuesta (0.350) está
    dominado por unos pocos benchmarks enormes, mientras que la métrica promedia
    los pares por igual (0.549 macro). Usar el primero como prior EMPEORA el
    Brier. Aquí el global y el efecto del sujeto se calculan macro, por par.

    Además el efecto del sujeto se contrae por el NÚMERO DE BENCHMARKS en los que
    aparece, no por el número de respuestas: lo que se pide es transferir a un
    benchmark nuevo, así que lo que informa es en cuántos benchmarks distintos se
    ha visto al sujeto, no cuántas veces se le ha preguntado en uno solo.
    """

    def __init__(self, fuerza: float = 4.0, k_sujeto_bench: float = 3.0):
        super().__init__(fuerza=fuerza)
        self.k_sujeto_bench = k_sujeto_bench

    def ajusta(self, tabla):
        t = tabla.copy()
        t["p"] = t["k"] / t["n"].clip(lower=1)
        g = float(t["p"].mean())                     # macro: cada par pesa igual
        self.global_logit = self._logit(g)
        t["d"] = t["p"].map(self._logit) - self.global_logit
        s = t.groupby("subject_id")["d"].agg(["sum", "count"])
        # contracción por número de benchmarks observados del sujeto
        self.ef_sujeto = (s["sum"] / (s["count"] + self.k_sujeto_bench)).to_dict()
        self.ef_bench = {}                           # bajo LOBO nunca se usa
        return self


# --------------------------------------------- Bayes empírico no paramétrico

class EmpiricoBayes:
    """Prior NO PARAMÉTRICO sobre la tasa de acierto del par, estimado fuera del
    benchmark objetivo, y media posterior exacta sobre esa rejilla.

    Motivación medida en este sprint: la distribución de tasas por par
    sujeto-benchmark es bimodal (≈7,7 % de pares por debajo de 0,05 y una joroba
    ancha hacia 0,78). Una Beta única, que es lo que implementa la contracción
    beta-binomial, no puede representar esa forma. Con una rejilla empírica:

        p̂ = Σ_j w_j π_j^(k+1) (1-π_j)^(n-k) / Σ_j w_j π_j^k (1-π_j)^(n-k)

    donde (π_j, w_j) es el histograma de tasas de los OTROS benchmarks. Con n=0
    devuelve la media del prior; con n grande converge a la media empírica.

    `mezcla_uniforme` evita pesos nulos y acota cuánto puede afirmar el prior.
    `min_n` restringe el prior a pares con suficientes respuestas, que son los
    únicos que el concurso evalúa (exige >= 80 ítems distintos por par).
    """

    def __init__(self, n_rejilla: int = 101, mezcla_uniforme: float = 0.05,
                 min_n: int = 80):
        self.n_rejilla, self.mezcla, self.min_n = n_rejilla, mezcla_uniforme, min_n
        self.grid = (np.arange(n_rejilla) + 0.5) / n_rejilla
        self.w = np.full(n_rejilla, 1.0 / n_rejilla)

    def ajusta(self, tabla):
        t = tabla[tabla["n"] >= self.min_n]
        if len(t) < 50:
            t = tabla
        p = (t["k"] / t["n"].clip(lower=1)).to_numpy()
        idx = np.clip((p * self.n_rejilla).astype(int), 0, self.n_rejilla - 1)
        w = np.bincount(idx, minlength=self.n_rejilla).astype(float)
        w /= w.sum()
        self.w = (1 - self.mezcla) * w + self.mezcla / self.n_rejilla
        return self

    def __call__(self, par, labeled_idx):
        n = len(labeled_idx)
        k = int(par.adq_y[labeled_idx].sum()) if n else 0
        g, w = self.grid, self.w
        # log-verosimilitud para evitar desbordamiento con n=31
        ll = k * np.log(g) + (n - k) * np.log1p(-g)
        ll -= ll.max()
        post = w * np.exp(ll)
        s = post.sum()
        p = float((post * g).sum() / s) if s > 0 else 0.5
        return np.full(len(par.eva_items), _clip(p))


class EmpiricoBayesSujeto(EmpiricoBayes):
    """Bayes empírico con el prior condicionado al sujeto cuando hay evidencia.

    Si el sujeto aparece en al menos `min_bench` benchmarks distintos fuera del
    objetivo, el prior se reconstruye con SUS tasas; si no, se usa el global.
    Comprueba si la identidad del sistema transfiere a un benchmark nuevo.
    """

    def __init__(self, min_bench: int = 4, **kw):
        super().__init__(**kw)
        self.min_bench = min_bench
        self.w_sujeto: dict[str, np.ndarray] = {}

    def ajusta(self, tabla):
        super().ajusta(tabla)
        t = tabla[tabla["n"] >= self.min_n]
        if len(t) < 50:
            t = tabla
        p = (t["k"] / t["n"].clip(lower=1)).to_numpy()
        idx = np.clip((p * self.n_rejilla).astype(int), 0, self.n_rejilla - 1)
        for sid, pos in pd_groupby_indices(t["subject_id"].to_numpy()):
            if len(pos) < self.min_bench:
                continue
            w = np.bincount(idx[pos], minlength=self.n_rejilla).astype(float)
            w /= w.sum()
            self.w_sujeto[sid] = (1 - self.mezcla) * w + self.mezcla / self.n_rejilla
        return self

    def __call__(self, par, labeled_idx):
        w0 = self.w
        self.w = self.w_sujeto.get(par.subject_id, self.w)
        try:
            return super().__call__(par, labeled_idx)
        finally:
            self.w = w0


def pd_groupby_indices(arr):
    """Índices por valor, sin construir un DataFrame."""
    orden = np.argsort(arr, kind="stable")
    ordenado = arr[orden]
    corte = np.flatnonzero(np.r_[True, ordenado[1:] != ordenado[:-1], True])
    return [(ordenado[corte[i]], orden[corte[i]:corte[i + 1]]) for i in range(len(corte) - 1)]
