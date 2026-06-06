from pathlib import Path
from preprocesado_de_datos.utils.path_utils import escribir_csv_dicts
import numpy as np

# funcion que calcula la diversidad normalizada
def calcular_diversidad_normalizada(poblacion):
    poblacion = np.asarray(poblacion, dtype=float)
    if poblacion.ndim != 2 or poblacion.shape[0] == 0 or poblacion.shape[1] <= 0:
        return float("nan")

    centroide = np.mean(poblacion, axis=0)
    distancias = np.linalg.norm(poblacion - centroide, axis=1)
    diversidad = float(np.mean(distancias))
    return diversidad / int(poblacion.shape[1])

# funcion que ordena la poblacion segun el valor de fitness de cada individuo y devuelve el mejor.
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
    ratio_estabilidad_diversidad=None,
    ratio_paciencia=None,
    reinicio_elitista=False,
):
    eventos = list(eventos or [])
    if ratio_paciencia is None and eventos:
        primer_evento = eventos[0]
        if primer_evento.get("paciencia_evals") not in (None, ""):
            max_evals = primer_evento.get("max_evals")
            if max_evals:
                ratio_paciencia = float(primer_evento["paciencia_evals"]) / float(max_evals)
    return {
        "reinicio_elitista": bool(reinicio_elitista),
        "reinicio_elitista_ratio_estabilidad_diversidad": (
            float(ratio_estabilidad_diversidad)
            if ratio_estabilidad_diversidad is not None
            else None
        ),
        "reinicio_elitista_ratio_paciencia": (
            float(ratio_paciencia) if ratio_paciencia is not None else None
        ),
        "reinicio_elitista_criterio": (
            "segundo_mejor_estancado"
            if reinicio_elitista
            else ""
        ),
        "n_reinicios_elitistas": int(len(eventos)),
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
        "ratio_estabilidad_diversidad",
        "ventana_evaluaciones",
        "div_dist_euclidea_normalizada",
        "mejor_fitness",
        "segundo_mejor_fitness",
        "evals_desde_mejora_mejor",
        "evals_desde_mejora_segundo",
        "paciencia_evals",
        "delta_diversidad_ventana",
        "media_diversidad_ventana",
        "ratio_delta_diversidad_ventana",
        "criterio_mejor_estancado",
        "criterio_segundo_estancado",
        "criterio_fitness_estancado",
        "criterio_diversidad_estable",
        "criterio_reinicio",
        "reinicio",
        "indice_individuo_preservado",
        "fitness_preservado",
    ]
    escribir_csv_dicts(ruta_csv, eventos, fieldnames=fieldnames)
    return str(ruta_csv)


class ControlReinicioElitista:
    """Controla reinicios elitistas por estancamiento del segundo mejor.

    El mejor individuo se conserva al reiniciar, por lo que no se usa como
    senal principal de estancamiento. El reinicio se activa cuando el segundo
    mejor fitness no mejora durante la paciencia configurada. La diversidad se
    calcula y se guarda solo como diagnostico.
    """

    def __init__(
        self,
        ratio_estabilidad_diversidad=None,
        *,
        max_evals=None,
        paciencia_evals=None,
        ratio_paciencia=0.05,
        tolerancia_fitness_abs=1e-6,
        tolerancia_fitness_rel=0.0,
        ventana_evaluaciones=2500,
    ):
        self.ratio_estabilidad_diversidad = (
            float(ratio_estabilidad_diversidad)
            if ratio_estabilidad_diversidad is not None
            else None
        )
        self.max_evals = int(max_evals) if max_evals is not None else None
        self.paciencia_evals = (
            int(paciencia_evals) if paciencia_evals is not None else None
        )
        self.ratio_paciencia = float(ratio_paciencia)
        self.tolerancia_fitness_abs = float(tolerancia_fitness_abs)
        self.tolerancia_fitness_rel = float(tolerancia_fitness_rel)
        self.ventana_evaluaciones = max(1, int(ventana_evaluaciones))
        self.reset(max_evals=max_evals)

    # reinicia el estado interno del controlador limpiando las referencias del mejor y segundo fitness, etc...
    def reset(self, *, max_evals=None):
        if max_evals is not None:
            self.max_evals = int(max_evals)
        self.mejor_referencia = None
        self.segundo_referencia = None
        self.eval_ultima_mejora_mejor = None
        self.eval_ultima_mejora_segundo = None
        self.historial_diversidad = []
        self.ultimo_diagnostico = {}

    # devuelve cuantas evaluaciones deben pasar sin mejora antes de considerar estancamiento
    def _paciencia_actual(self):
        if self.paciencia_evals is not None:
            return max(1, int(self.paciencia_evals))
        if self.max_evals is not None:
            return max(1, int(round(float(self.max_evals) * self.ratio_paciencia)))
        return 5000

    # comprueba si un valor fitness actual mejora de forma real a una referencia anterior. para no considerar
    # como mejora cambios insignificantes:
    #   - tolerancia_abs: margen minimo fijo
    #   - tolerancia_rel: margen proporcional al tamaño del fitness
    def _hay_mejora(self, valor_actual, valor_referencia):
        if valor_referencia is None:
            return True
        margen = max(
            self.tolerancia_fitness_abs,
            abs(float(valor_referencia)) * self.tolerancia_fitness_rel,
        )
        return float(valor_actual) < float(valor_referencia) - margen

    # ordena el vector de fitness y devuelve el mejor y el segundo mejor valor. 
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

    # comprueba si la diversidad apenas cambia dentro de la ventana de evaluaciones reciente.
    # el historial ya esta recortado a la ventana en debe_reiniciar, asi que no es necesario filtrar.
    def _diversidad_estable(self):
        if self.ratio_estabilidad_diversidad is None:
            return False, float("nan"), float("nan"), float("nan")
        if len(self.historial_diversidad) < 2:
            return False, float("nan"), float("nan"), float("nan")
        ventana = np.asarray([d for _, d in self.historial_diversidad], dtype=float)
        ventana = ventana[np.isfinite(ventana)]
        if ventana.size < 2:
            return False, float("nan"), float("nan"), float("nan")
        delta = float(np.max(ventana) - np.min(ventana))
        media = float(np.mean(np.abs(ventana)))
        escala = max(media, 1e-12)
        ratio_delta = float(delta / escala)
        estable = bool(ratio_delta <= self.ratio_estabilidad_diversidad)
        return estable, delta, media, ratio_delta

    # actualiza las referencias de mejor y segundo mejor fitness justo despues de aplicar un reinicio. evita que el controlador vuelva a reinicar inmediatamente usando referencias antiguas
    def registrar_estado_post_reinicio(self, fitness, evaluaciones):
        mejor, segundo = self._mejor_y_segundo(fitness)
        evaluaciones = int(evaluaciones)
        if np.isfinite(mejor):
            self.mejor_referencia = mejor
            self.eval_ultima_mejora_mejor = evaluaciones
        if np.isfinite(segundo):
            self.segundo_referencia = segundo
            self.eval_ultima_mejora_segundo = evaluaciones

    # calcula el mejor fitness, segundo mejor fitness, diversidad normalizada,
    # evals desde la ultima mejora del mejor y segundo mejor. El reinicio se
    # decide por el segundo mejor; la diversidad queda registrada como
    # diagnostico.
    def debe_reiniciar(self, fitness, poblacion, evaluaciones, generacion, bounds):
        evaluaciones = int(evaluaciones)
        mejor, segundo = self._mejor_y_segundo(fitness)
        diversidad_norm = calcular_diversidad_normalizada(poblacion)

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

        self.historial_diversidad.append((evaluaciones, float(diversidad_norm)))
        # recortar entradas fuera de la ventana para mantener el historial acotado
        cutoff = evaluaciones - self.ventana_evaluaciones
        while self.historial_diversidad and self.historial_diversidad[0][0] < cutoff:
            self.historial_diversidad.pop(0)
        (
            criterio_diversidad_estable,
            delta_diversidad,
            media_diversidad,
            ratio_delta_diversidad,
        ) = self._diversidad_estable()
        reiniciar = bool(criterio_segundo_estancado)

        self.ultimo_diagnostico = {
            "generacion": int(generacion),
            "evaluaciones": int(evaluaciones),
            "mejor_fitness": float(mejor),
            "segundo_mejor_fitness": float(segundo),
            "evals_desde_mejora_mejor": int(evals_desde_mejor),
            "evals_desde_mejora_segundo": int(evals_desde_segundo),
            "paciencia_evals": int(paciencia),
            "ratio_estabilidad_diversidad": (
                float(self.ratio_estabilidad_diversidad)
                if self.ratio_estabilidad_diversidad is not None
                else None
            ),
            "ventana_evaluaciones": int(self.ventana_evaluaciones),
            "div_dist_euclidea_normalizada": float(diversidad_norm),
            "delta_diversidad_ventana": float(delta_diversidad),
            "media_diversidad_ventana": float(media_diversidad),
            "ratio_delta_diversidad_ventana": float(ratio_delta_diversidad),
            "criterio_mejor_estancado": bool(criterio_mejor_estancado),
            "criterio_segundo_estancado": bool(criterio_segundo_estancado),
            "criterio_fitness_estancado": bool(criterio_fitness_estancado),
            "criterio_diversidad_estable": bool(criterio_diversidad_estable),
            "criterio_reinicio": "segundo_mejor_estancado",
            "reinicio": bool(reiniciar),
        }
        return reiniciar

    def diagnostico_reinicio(self):
        return dict(self.ultimo_diagnostico)
