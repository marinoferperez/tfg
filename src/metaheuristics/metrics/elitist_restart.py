"""
Utilidades de reinicio elitista para metaheurísticas de evolución diferencial y AGE.

Implementa la política de reinicio por estancamiento del segundo mejor individuo:
cuando el segundo mejor fitness no mejora durante la ventana de estancamiento, se
regenera la población conservando únicamente el mejor individuo (elite).
"""

import numpy as np

def seleccionar_indice_elitista(fitness):
    """
    Devuelve el índice del individuo con mejor (menor) fitness.

    fitness: vector de valores de fitness de la población actual.
    """
    valores = np.asarray(fitness, dtype=float)
    if valores.ndim != 1:
        raise ValueError("fitness debe ser un vector unidimensional.")
    if valores.size == 0:
        raise ValueError("fitness no puede estar vacio.")
    orden = np.argsort(valores, kind="stable")
    return int(orden[0])


def construir_metadata_reinicios(eventos, ratio_estancamiento=None, reinicio=False):
    """
    Construye los metadatos de reinicio para incluir en el resumen JSON.

    eventos: lista de dicts devueltos por ControlReinicioElitista.diagnostico_reinicio().
    ratio_estancamiento: fracción de max_evals usada como ventana de estancamiento.
    reinicio: indica si el mecanismo de reinicio estaba activo.
    """
    eventos = list(eventos or [])
    return {
        "reinicio": bool(reinicio),
        "reinicio_ratio": (
            float(ratio_estancamiento) if ratio_estancamiento is not None else None
        ),
        "reinicio_criterio": "segundo_mejor_estancado" if reinicio else "",
        "n_reinicios": int(len(eventos)),
        "generaciones_reinicio": [int(evento["generacion"]) for evento in eventos],
        "evaluaciones_reinicio": [int(evento["evals_despues_reinicio"]) for evento in eventos],
    }

class ControlReinicioElitista:
    """Controla reinicios elitistas por estancamiento del segundo mejor.

    El mejor individuo se conserva al reiniciar. El reinicio se activa cuando el segundo mejor fitness no mejora durante la estancamiento configurada.
    """

    def __init__(self, *, max_evals=None, estancamiento_evals=None, ratio_estancamiento=0.05, tolerancia_fitness_abs=1e-6):
        """
        Configura el controlador de reinicio elitista.

        max_evals: presupuesto total de evaluaciones.
        estancamiento_evals: número fijo de evaluaciones sin mejora antes de reiniciar.
        ratio_estancamiento: fracción de max_evals que sirve de estancamiento si no se fija estancamiento_evals directamente.
        tolerancia_fitness_abs: mejora absoluta mínima requerida para resetear el contador de estancamiento.
        """
        self.max_evals = int(max_evals) if max_evals is not None else None
        self.estancamiento_evals = (int(estancamiento_evals) if estancamiento_evals is not None else None)
        self.ratio_estancamiento = float(ratio_estancamiento)
        self.tolerancia_fitness_abs = float(tolerancia_fitness_abs)
        self.reset(max_evals=max_evals)

    def reset(self, *, max_evals=None):
        """
        Reinicia el estado interno del controlador.

        max_evals: si se indica, actualiza el presupuesto total antes de limpiar.
        """
        if max_evals is not None:
            self.max_evals = int(max_evals)
        self.mejor_referencia = None
        self.segundo_referencia = None
        self.eval_ultima_mejora_mejor = None
        self.eval_ultima_mejora_segundo = None
        self.ultimo_diagnostico = {}

    def _estancamiento_actual(self):
        """Calcula el estancamiento en evaluaciones efectiva para la ejecución actual."""
        if self.estancamiento_evals is not None:
            return max(1, int(self.estancamiento_evals))
        if self.max_evals is not None:
            return max(1, int(round(float(self.max_evals) * self.ratio_estancamiento)))
        return 5000

    def _hay_mejora(self, valor_actual, valor_referencia):
        """
        Comprueba si valor_actual mejora significativamente sobre la referencia.

        valor_actual: fitness actual del mejor o segundo mejor individuo.
        valor_referencia: fitness de referencia guardado en la última mejora registrada.

        La mejora debe superar tolerancia_fitness_abs para resetear el contador.
        """
        if valor_referencia is None:
            return True
        return float(valor_actual) < float(valor_referencia) - self.tolerancia_fitness_abs

    @staticmethod
    def _mejor_y_segundo(fitness):
        """
        Extrae el mejor y el segundo mejor fitness de la población.

        fitness: vector de valores de fitness de la población actual.
        """
        valores = np.asarray(fitness, dtype=float).reshape(-1)
        valores = valores[np.isfinite(valores)]
        if valores.size == 0:
            return float("nan"), float("nan")
        orden = np.sort(valores, kind="stable")
        mejor = float(orden[0])
        segundo = float(orden[1]) if orden.size > 1 else mejor
        return mejor, segundo

    def registrar_estado_post_reinicio(self, fitness, evaluaciones):
        """
        Actualiza las referencias de fitness tras un reinicio para evitar un reinicio inmediato en la siguiente comprobación.

        fitness: fitness de la nueva población generada tras el reinicio.
        evaluaciones: número de evaluaciones consumidas tras el reinicio.
        """
        mejor, segundo = self._mejor_y_segundo(fitness)
        evaluaciones = int(evaluaciones)
        if np.isfinite(mejor):
            self.mejor_referencia = mejor
            self.eval_ultima_mejora_mejor = evaluaciones
        if np.isfinite(segundo):
            self.segundo_referencia = segundo
            self.eval_ultima_mejora_segundo = evaluaciones

    def debe_reiniciar(self, fitness, evaluaciones, generacion):
        """
        Evalúa si se debe ejecutar un reinicio elitista en este punto.

        fitness: vector de fitness de la población actual.
        evaluaciones: número de evaluaciones consumidas hasta ahora.
        generacion: generación actual.

        Retorna True si el segundo mejor lleva sin mejorar más de la estancamiento.
        """
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
        estancamiento = self._estancamiento_actual()

        criterio_mejor_estancado = bool(evals_desde_mejor >= estancamiento)
        criterio_segundo_estancado = bool(evals_desde_segundo >= estancamiento)
        reiniciar = criterio_segundo_estancado

        self.ultimo_diagnostico = {
            "generacion": int(generacion),
            "evaluaciones": int(evaluaciones),
            "mejor_fitness": float(mejor),
            "segundo_mejor_fitness": float(segundo),
            "evals_desde_mejora_mejor": int(evals_desde_mejor),
            "evals_desde_mejora_segundo": int(evals_desde_segundo),
            "estancamiento_evals": int(estancamiento),
            "criterio_mejor_estancado": criterio_mejor_estancado,
            "criterio_segundo_estancado": criterio_segundo_estancado,
            # alias de criterio_segundo_estancado; se mantiene porque experiment_io lo lee para el CSV
            "criterio_fitness_estancado": criterio_segundo_estancado,
            "criterio_reinicio": "segundo_mejor_estancado",
            "reinicio": reiniciar,
        }
        return reiniciar

    def diagnostico_reinicio(self):
        """Devuelve una copia del diagnóstico generado en la última llamada a debe_reiniciar."""
        return dict(self.ultimo_diagnostico)
