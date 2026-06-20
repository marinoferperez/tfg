"""
Controlador de la integracion online con modelos subrogados.

Logica comun a AGE, DE y SHADE:
    - calentamiento inicial;
    - ventana no acumulativa de entrenamiento;
    - probabilidad de uso del subrogado;
    - entrenamiento del modelo;
    - prediccion;
    - aplicacion de la politica de rechazo;
    - registro de estadisticas.

El controlador solo decide si un candidato debe pasar a evaluacion real o puede rechazarse antes.
"""

from dataclasses import dataclass, field
import time
import numpy as np

from src.surrogates.select_model import select_model
from src.metaheuristics.surrogate.surrogate_policy import PoliticaSubrogado, DecisionSubrogado
from src.metaheuristics.surrogate.surrogate_stats import EstadisticasSubrogado

@dataclass(frozen=True)
class ConfiguracionSubrogadoOnline:
    """
    Configuracion general del subrogado online.

    warmup_ratio: porcentaje inicial del presupuesto evaluado solo con la funcion real.
    window_ratio: porcentaje del presupuesto usado como ventana reciente de entrenamiento (mismo porcentaje que el utilizado en la evaluacion offline).
    probabilidad_subrogado: probabilidad p de aplicar el filtro subrogado a un candidato.
    modelo_nombre / modelo_params: modelo construido mediante select_model.
    cooldown_reinicio_evals: número de evaluaciones reales que se esperan tras un reinicio antes de volver a usar el subrogado.
    retrain_ratio: fraccion de la ventana de entrenamiento que debe renovarse antes de reentrenar.
    """

    modelo_nombre: str = "rbf"
    modelo_params: dict = field(default_factory=lambda: {
        "kernel": "multiquadric",
        "epsilon": 1.0,
        "smoothing": 1e-3,
        "neighbors": 50,
        "degree": -1,
    })

    cooldown_reinicio_evals: int = 0
    warmup_ratio: float = 0.20
    window_ratio: float = 0.20
    probabilidad_subrogado: float = 0.50
    retrain_ratio: float = 0.25

    max_evals: int = 100000
    minimizacion: bool = True
    seed: int | None = None

    def __post_init__(self):
        """Valida que todos los parámetros de configuración son coherentes."""
        if not 0.0 <= self.warmup_ratio <= 1.0:
            raise ValueError("warmup_ratio debe estar en [0, 1].")
        if not 0.0 < self.window_ratio <= 1.0:
            raise ValueError("window_ratio debe estar en (0, 1].")
        if not 0.0 <= self.probabilidad_subrogado <= 1.0:
            raise ValueError("probabilidad_subrogado debe estar en [0, 1].")
        if int(self.max_evals) <= 0:
            raise ValueError("max_evals debe ser positivo.")
        if int(self.cooldown_reinicio_evals) < 0:
            raise ValueError("cooldown_reinicio_evals debe ser >= 0.")
        if not 0.0 < self.retrain_ratio <= 1.0:
            raise ValueError("retrain_ratio debe estar en (0, 1].")

@dataclass(frozen=True)
class ResultadoFiltroSubrogado:
    """
    Resultado devuelto a la metaheuristica.

    debe_evaluar: True si el candidato debe evaluarse con la funcion objetivo real.
    uso_subrogado: True si la decision se ha tomado usando prediccion del modelo.
    decision: Decision devuelta por la politica, si se aplico subrogado.
    motivo: cadena que identifica la causa de la decision (para estadisticas y CSV).
    """

    debe_evaluar: bool
    uso_subrogado: bool
    decision: DecisionSubrogado | None
    motivo: str

class ControladorSubrogadoOnline:
    """
    Controlador común para la hibridacion online.

    La metaheuristica llama a:
        - registrar_evaluacion_directa / registrar_evaluacion_tras_subrogado tras cada fitness real.
        - decidir_evaluacion(...) antes de evaluar un candidato opcionalmente filtrable.
    """

    def __init__(self, config, estadisticas=None):
        """
        config: ConfiguracionSubrogadoOnline con todos los hiperparámetros del subrogado.
        estadisticas: EstadisticasSubrogado opcional. Si es None se crea uno nuevo.
        """
        self.config = config
        self.estadisticas = estadisticas if estadisticas is not None else EstadisticasSubrogado()
        self.politica = PoliticaSubrogado(minimizacion=config.minimizacion)
        self.rng = np.random.default_rng(config.seed)

        # historial acumulado de vectores evaluados realmente; la ventana se extrae de los últimos window_size
        self._x_reales = []
        self._y_reales = []
        # el modelo se inicializa a None y se entrena en la primera llamada a decidir_evaluacion con subrogado activo
        self._modelo = None
        # número de evaluaciones reales que había cuando se entrenó el modelo por última vez
        self._modelo_entrenado_con_n = 0
        # evaluación real en que ocurrió el último reinicio; None si nunca hubo reinicio
        self._evals_ultimo_reinicio = None

    @property
    def evals_reales(self):
        """Número de evaluaciones reales de la función objetivo acumuladas hasta ahora."""
        return len(self._y_reales)

    @property
    def warmup_evals(self):
        """Número de evaluaciones reales requeridas antes de activar el subrogado."""
        return int(np.ceil(self.config.warmup_ratio * self.config.max_evals))

    @property
    def window_size(self):
        """Número de evaluaciones recientes usadas como ventana de entrenamiento."""
        return max(1, int(np.ceil(self.config.window_ratio * self.config.max_evals)))

    @property
    def retrain_interval_efectivo(self):
        """Nuevas evaluaciones reales que deben acumularse antes de reentrenar."""
        return max(1, int(np.ceil(self.config.retrain_ratio * self.window_size)))

    def puede_usar_subrogado(self):
        """Comprueba si ya existe informacion suficiente para activar el filtro."""
        if self.config.probabilidad_subrogado <= 0.0:
            return False
        if self.evals_reales < self.warmup_evals:
            return False

        # el cooldown bloquea el subrogado un número fijo de evaluaciones tras cada reinicio
        if self._evals_ultimo_reinicio is not None and self.config.cooldown_reinicio_evals > 0:
            evals_desde_reinicio = self.evals_reales - self._evals_ultimo_reinicio
            if evals_desde_reinicio < self.config.cooldown_reinicio_evals:
                return False

        return True

    def decidir_evaluacion(self, candidato, fitness_ref, generacion=None):
        """
        Decide si un candidato debe evaluarse realmente.

        candidato: vector de decision a filtrar.
        fitness_ref: fitness real de referencia (padre en DE/SHADE, peor de la población en AGE).
        generacion: generación actual, usada solo para el registro de estadísticas.

        La metaheuristica debe llamar a este metodo antes de gastar una
        evaluacion real en un candidato que pueda filtrarse.

        Retorna un ResultadoFiltroSubrogado con la decision y su motivo.
        """
        self.estadisticas.registrar_candidato_generado()

        if not self.puede_usar_subrogado():
            return ResultadoFiltroSubrogado(
                debe_evaluar=True,
                uso_subrogado=False,
                decision=None,
                motivo="subrogado_no_activo",
            )

        # muestreo de Bernoulli: con probabilidad (1 - p) el candidato se evalúa directamente sin filtro
        if self.rng.random() >= self.config.probabilidad_subrogado:
            return ResultadoFiltroSubrogado(
                debe_evaluar=True,
                uso_subrogado=False,
                decision=None,
                motivo="evaluacion_directa_por_probabilidad",
            )

        self.estadisticas.registrar_candidato_con_subrogado()
        self._asegurar_modelo_entrenado()

        # predicción puntual del fitness del candidato con el modelo actual
        inicio_pred = time.perf_counter()
        prediccion = self._modelo.predict(np.asarray(candidato, dtype=float).reshape(1, -1))
        tiempo_pred = time.perf_counter() - inicio_pred
        self.estadisticas.registrar_prediccion(tiempo_pred)

        decision = self.politica.decidir(
            fitness_pred=prediccion,
            fitness_ref=fitness_ref,
        )
        self._registrar_decision_subrogado(decision, generacion=generacion)

        if decision.debe_evaluar:
            self.estadisticas.registrar_aceptado_por_subrogado()
            return ResultadoFiltroSubrogado(
                debe_evaluar=True,
                uso_subrogado=True,
                decision=decision,
                motivo=decision.motivo,
            )

        self.estadisticas.registrar_rechazo_subrogado()
        return ResultadoFiltroSubrogado(
            debe_evaluar=False,
            uso_subrogado=True,
            decision=decision,
            motivo=decision.motivo,
        )

    def _registrar_decision_subrogado(self, decision, generacion=None):
        """
        Registra la decisión del subrogado en el historial de estadísticas.

        decision: DecisionSubrogado devuelta por la politica.
        generacion: generación actual, o None si no se dispone de ella.
        """
        pred = decision.fitness_pred
        ref = decision.fitness_ref
        # margen positivo → el subrogado predice que el candidato es peor que la referencia
        margen = None if pred is None or ref is None else float(pred) - float(ref)
        evals_desde_reinicio = (
            None
            if self._evals_ultimo_reinicio is None
            else int(self.evals_reales - self._evals_ultimo_reinicio)
        )

        self.estadisticas.registrar_decision_subrogado(
            evals_reales=int(self.evals_reales),
            generacion=None if generacion is None else int(generacion),
            reinicios=int(self.estadisticas.reinicios),
            evals_desde_reinicio=evals_desde_reinicio,
            fitness_pred=pred,
            fitness_ref=ref,
            margen_pred_ref=margen,
            debe_evaluar=bool(decision.debe_evaluar),
            motivo=str(decision.motivo),
        )

    def registrar_evaluacion_directa(self, x, fitness):
        """
        Registra una evaluacion real que no paso por el filtro subrogado.

        x: vector de decision evaluado.
        fitness: valor real de la funcion objetivo para x.

        Permite distinguir en estadisticas entre evaluaciones directas y las
        permitidas por el filtro del subrogado.
        """
        self._x_reales.append(np.asarray(x, dtype=float).copy())
        self._y_reales.append(float(fitness))
        self.estadisticas.registrar_evaluacion_directa()

    def registrar_evaluacion_tras_subrogado(self, x, fitness):
        """
        Registra una evaluacion real realizada tras pasar el filtro subrogado.

        x: vector de decision evaluado.
        fitness: valor real de la funcion objetivo para x.
        """
        self._x_reales.append(np.asarray(x, dtype=float).copy())
        self._y_reales.append(float(fitness))
        self.estadisticas.registrar_evaluacion_tras_subrogado()

    def registrar_reinicio(self):
        """
        Registra un reinicio de la metaheuristica.

        No se borra el historico de evaluaciones. La estrategia no acumulativa se
        aplica seleccionando siempre la ventana reciente en el entrenamiento.
        El modelo se invalida para forzar reentrenamiento con datos post-reinicio.
        """
        self.estadisticas.registrar_reinicio()

        # el modelo queda obsoleto tras el reinicio: la población cambió completamente
        self._modelo = None
        self._modelo_entrenado_con_n = 0
        self._evals_ultimo_reinicio = int(self.evals_reales)

    def _asegurar_modelo_entrenado(self):
        """
        Entrena el modelo si no existe o si han entrado suficientes evaluaciones
        reales nuevas desde el ultimo entrenamiento.
        """
        if self._modelo is not None:
            evals_desde_entrenamiento = self.evals_reales - self._modelo_entrenado_con_n
            # si no han llegado suficientes evaluaciones nuevas, el modelo actual sigue siendo válido
            if evals_desde_entrenamiento < self.retrain_interval_efectivo:
                return

        x_train, y_train = self._ventana_entrenamiento()

        inicio_train = time.perf_counter()
        modelo = select_model(
            self.config.modelo_nombre,
            **dict(self.config.modelo_params),
        )
        modelo.fit(x_train, y_train)
        tiempo_train = time.perf_counter() - inicio_train

        self._modelo = modelo
        self._modelo_entrenado_con_n = self.evals_reales
        self.estadisticas.registrar_entrenamiento(tiempo_train)

    def _ventana_entrenamiento(self):
        """Devuelve la ventana no acumulativa usada para entrenar el subrogado."""
        # se toman los últimos window_size puntos; si hay menos, se usan todos
        n = min(self.window_size, self.evals_reales)
        x_train = np.asarray(self._x_reales[-n:], dtype=float)
        y_train = np.asarray(self._y_reales[-n:], dtype=float)
        return x_train, y_train

    def resumen(self):
        """Devuelve las estadisticas agregadas junto con la configuracion usada."""
        resumen = self.estadisticas.resumen()
        resumen.update(
            {
                "modelo_nombre": self.config.modelo_nombre,
                "modelo_params": dict(self.config.modelo_params),
                "warmup_ratio": float(self.config.warmup_ratio),
                "window_ratio": float(self.config.window_ratio),
                "probabilidad_subrogado": float(self.config.probabilidad_subrogado),
                "max_evals": int(self.config.max_evals),
                "warmup_evals": int(self.warmup_evals),
                "window_size": int(self.window_size),
                "cooldown_reinicio_evals": int(self.config.cooldown_reinicio_evals),
                "retrain_ratio": float(self.config.retrain_ratio),
                "retrain_interval_efectivo": int(self.retrain_interval_efectivo),
            }
        )
        return resumen
