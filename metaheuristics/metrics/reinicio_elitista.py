from pathlib import Path
from preprocesado_de_datos.utils.path_utils import escribir_csv_dicts
import numpy as np


def seleccionar_indice_elitista(fitness):
    valores = np.asarray(fitness, dtype=float)
    if valores.ndim != 1:
        raise ValueError("fitness debe ser un vector unidimensional.")
    if valores.size == 0:
        raise ValueError("fitness no puede estar vacio.")
    orden = np.argsort(valores, kind="stable")
    return int(orden[0])


def construir_metadata_reinicios(
    eventos,
    ratio_paciencia=None,
    reinicio=False,
):
    eventos = list(eventos or [])
    if ratio_paciencia is None and eventos:
        primer_evento = eventos[0]
        if primer_evento.get("paciencia_evals") not in (None, ""):
            max_evals = primer_evento.get("max_evals")
            if max_evals:
                ratio_paciencia = float(primer_evento["paciencia_evals"]) / float(max_evals)
    return {
        "reinicio": bool(reinicio),
        "reinicio_ratio": (
            float(ratio_paciencia) if ratio_paciencia is not None else None
        ),
        "reinicio_criterio": (
            "segundo_mejor_estancado"
            if reinicio
            else ""
        ),
        "n_reinicios": int(len(eventos)),
        "generaciones_reinicio": [int(evento["generacion"]) for evento in eventos],
        "evaluaciones_reinicio": [
            int(evento["evaluaciones_despues_reinicio"])
            for evento in eventos
        ],
    }


def guardar_reinicios_elitistas_csv(ruta_base, eventos):
    eventos = list(eventos or [])
    if not eventos:
        return None

    ruta_csv = Path(ruta_base) / "reinicios_elitistas.csv"
    fieldnames = [
        "generacion",
        "evaluaciones_antes_reinicio",
        "evaluaciones_despues_reinicio",
        "mejor_fitness",
        "segundo_mejor_fitness",
        "evals_desde_mejora_mejor",
        "evals_desde_mejora_segundo",
        "paciencia_evals",
        "criterio_mejor_estancado",
        "criterio_segundo_estancado",
        "criterio_fitness_estancado",
        "criterio_reinicio",
        "reinicio",
        "indice_individuo_preservado",
        "fitness_preservado",
    ]
    escribir_csv_dicts(ruta_csv, eventos, fieldnames=fieldnames)
    return str(ruta_csv)


class ControlReinicioElitista:
    """Controla reinicios elitistas por estancamiento del segundo mejor.

    El mejor individuo se conserva al reiniciar. El reinicio se activa cuando
    el segundo mejor fitness no mejora durante la paciencia configurada.
    """

    def __init__(
        self,
        *,
        max_evals=None,
        paciencia_evals=None,
        ratio_paciencia=0.05,
        tolerancia_fitness_abs=1e-6,
        tolerancia_fitness_rel=0.0,
    ):
        self.max_evals = int(max_evals) if max_evals is not None else None
        self.paciencia_evals = (
            int(paciencia_evals) if paciencia_evals is not None else None
        )
        self.ratio_paciencia = float(ratio_paciencia)
        self.tolerancia_fitness_abs = float(tolerancia_fitness_abs)
        self.tolerancia_fitness_rel = float(tolerancia_fitness_rel)
        self.reset(max_evals=max_evals)

    def reset(self, *, max_evals=None):
        if max_evals is not None:
            self.max_evals = int(max_evals)
        self.mejor_referencia = None
        self.segundo_referencia = None
        self.eval_ultima_mejora_mejor = None
        self.eval_ultima_mejora_segundo = None
        self.ultimo_diagnostico = {}

    def _paciencia_actual(self):
        if self.paciencia_evals is not None:
            return max(1, int(self.paciencia_evals))
        if self.max_evals is not None:
            return max(1, int(round(float(self.max_evals) * self.ratio_paciencia)))
        return 5000

    def _hay_mejora(self, valor_actual, valor_referencia):
        if valor_referencia is None:
            return True
        margen = max(
            self.tolerancia_fitness_abs,
            abs(float(valor_referencia)) * self.tolerancia_fitness_rel,
        )
        return float(valor_actual) < float(valor_referencia) - margen

    @staticmethod
    def _mejor_y_segundo(fitness):
        valores = np.asarray(fitness, dtype=float).reshape(-1)
        valores = valores[np.isfinite(valores)]
        if valores.size == 0:
            return float("nan"), float("nan")
        orden = np.sort(valores, kind="stable")
        mejor = float(orden[0])
        segundo = float(orden[1]) if orden.size > 1 else mejor
        return mejor, segundo

    def registrar_estado_post_reinicio(self, fitness, evaluaciones):
        mejor, segundo = self._mejor_y_segundo(fitness)
        evaluaciones = int(evaluaciones)
        if np.isfinite(mejor):
            self.mejor_referencia = mejor
            self.eval_ultima_mejora_mejor = evaluaciones
        if np.isfinite(segundo):
            self.segundo_referencia = segundo
            self.eval_ultima_mejora_segundo = evaluaciones

    def debe_reiniciar(self, fitness, evaluaciones, generacion):
        evaluaciones = int(evaluaciones)
        mejor, segundo = self._mejor_y_segundo(fitness)

        mejora_mejor = self._hay_mejora(mejor, self.mejor_referencia)
        mejora_segundo = self._hay_mejora(segundo, self.segundo_referencia)

        if mejora_mejor and np.isfinite(mejor):
            self.mejor_referencia = mejor
            self.eval_ultima_mejora_mejor = evaluaciones
        if mejora_segundo and np.isfinite(segundo):
            self.segundo_referencia = segundo
            self.eval_ultima_mejora_segundo = evaluaciones

        if self.eval_ultima_mejora_mejor is None:
            self.eval_ultima_mejora_mejor = evaluaciones
        if self.eval_ultima_mejora_segundo is None:
            self.eval_ultima_mejora_segundo = evaluaciones

        evals_desde_mejor = int(evaluaciones - self.eval_ultima_mejora_mejor)
        evals_desde_segundo = int(evaluaciones - self.eval_ultima_mejora_segundo)
        paciencia = self._paciencia_actual()

        criterio_mejor_estancado = bool(evals_desde_mejor >= paciencia)
        criterio_segundo_estancado = bool(evals_desde_segundo >= paciencia)
        criterio_fitness_estancado = bool(criterio_segundo_estancado)
        reiniciar = bool(criterio_segundo_estancado)

        self.ultimo_diagnostico = {
            "generacion": int(generacion),
            "evaluaciones": int(evaluaciones),
            "mejor_fitness": float(mejor),
            "segundo_mejor_fitness": float(segundo),
            "evals_desde_mejora_mejor": int(evals_desde_mejor),
            "evals_desde_mejora_segundo": int(evals_desde_segundo),
            "paciencia_evals": int(paciencia),
            "criterio_mejor_estancado": bool(criterio_mejor_estancado),
            "criterio_segundo_estancado": bool(criterio_segundo_estancado),
            "criterio_fitness_estancado": bool(criterio_fitness_estancado),
            "criterio_reinicio": "segundo_mejor_estancado",
            "reinicio": bool(reiniciar),
        }
        return reiniciar

    def diagnostico_reinicio(self):
        return dict(self.ultimo_diagnostico)
