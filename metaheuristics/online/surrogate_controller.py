# Debe ser el núcleo de la integración online. Sus responsabilidades:

# - almacenar las evaluaciones reales disponibles;
# - decidir cuándo el subrogado puede activarse;
# - entrenar RBF con la ventana reciente;
# - lanzar predicciones;
# - aplicar la política de rechazo/aceptación;
# - acumular tiempos de entrenamiento y predicción;
# - registrar contadores agregados.

# Parámetros principales:

# ```text
# warmup_ratio = 0.20
# window_ratio = 0.20
# surrogate_probability = p
# max_evals = 100000
# population_size = 50
# ```

# El subrogado se activa solo cuando:

# ```text
# evals_reales >= warmup_ratio * max_evals
# ```

# En nuestro caso:

# ```text
# evals_reales >= 20000
# ```

"""
Controlador de la integracion online con modelos subrogados.

Este modulo centraliza la logica comun a AGE, DE y SHADE:
    - calentamiento inicial;
    - ventana no acumulativa de entrenamiento;
    - probabilidad de uso del subrogado;
    - entrenamiento del modelo;
    - prediccion;
    - aplicacion de la politica de rechazo;
    - registro de estadisticas.

Las metaheuristicas siguen siendo responsables de generar candidatos y aplicar
su logica de reemplazo. El controlador solo decide si un candidato debe pasar a
evaluacion real o puede rechazarse antes.
"""

from dataclasses import dataclass, field
import time
import numpy as np

from surrogate_models.select_model import select_model
from metaheuristics.online.surrogate_policy import PoliticaSubrogado, DecisionSubrogado
from metaheuristics.online.surrogate_stats import EstadisticasSubrogado


@dataclass(frozen=True)
class ConfiguracionSubrogadoOnline:
    """
    Configuracion general del subrogado online.

    warmup_ratio:
        Porcentaje inicial del presupuesto evaluado solo con la funcion real.

    window_ratio:
        Porcentaje del presupuesto usado como ventana reciente de entrenamiento.

    probabilidad_subrogado:
        Probabilidad p de aplicar el filtro subrogado a un candidato.

    modelo_nombre / modelo_params:
        Modelo construido mediante surrogate_models.select_model.select_model.
    """
    
    modelo_nombre: str = "rbf"
    modelo_params: dict = field(default_factory=lambda: {
        "kernel": "multiquadric",
        "epsilon": 1.0,
        "smoothing": 1e-3,
        "neighbors": 50,
        "degree": -1,
    })
    
    # número de evaluaciones reales que se esperan tras un reinicio antes de volver a usar el subrogado.
    cooldown_reinicio_evals: int = 0

    # porcentaje inicial del presupuesto en el que el subrogado esta apagado
    warmup_ratio: float = 0.20
    
    # tamaño de la ventana reciente usada para entrenar el subrogado (mismo porcentaje que el utilizado en la evaluacion offline)
    window_ratio: float = 0.20
    
    # probabilida de aplicar el filtro subrogado a cada candidato
    probabilidad_subrogado: float = 0.50 
    
    # Fraccion de la ventana de entrenamiento que debe renovarse antes de reentrenar.
    # Por ejemplo, con window_size=2000 y retrain_ratio=0.25, se reentrena cada 500 evaluaciones reales.
    retrain_ratio: float = 0.25
    
    max_evals: int = 100000
    minimizacion: bool = True
    seed: int | None = None
    
    def __post_init__(self):
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

    debe_evaluar:
        True si el candidato debe evaluarse con la funcion objetivo real.

    uso_subrogado:
        True si la decision se ha tomado usando prediccion del modelo.

    decision:
        Decision devuelta por la politica, si se aplico subrogado.
    """
    
    debe_evaluar: bool
    uso_subrogado: bool
    decision: DecisionSubrogado | None
    motivo: str

class ControladorSubrogadoOnline:
    """
    Controlador comun para la hibridacion online.

    La metaheuristica debe llamar a:
        - registrar_evaluacion_real(...) cada vez que evalua la funcion real;
        - decidir_evaluacion(...) antes de evaluar un candidato opcionalmente filtrable.
    """
    
    def __init__(self, config, estadisticas=None):
        self.config = config
        self.estadisticas = estadisticas if estadisticas is not None else EstadisticasSubrogado()
        self.politica = PoliticaSubrogado(minimizacion=config.minimizacion)
        self.rng = np.random.default_rng(config.seed)

        self._x_reales = []
        self._y_reales = []
        self._modelo = None
        self._modelo_entrenado_con_n = 0
        self._evals_ultimo_reinicio = None
        
    @property
    def evals_reales(self):
        return len(self._y_reales)

    @property
    def warmup_evals(self):
        return int(np.ceil(self.config.warmup_ratio * self.config.max_evals))

    @property
    def window_size(self):
        return max(1, int(np.ceil(self.config.window_ratio * self.config.max_evals)))

    @property
    def retrain_interval_efectivo(self):
        return max(1, int(np.ceil(self.config.retrain_ratio * self.window_size)))
    
    def registrar_evaluacion_real(self, x, fitness) -> None:
        """
        Registra una evaluacion real disponible para futuros entrenamientos.

        Esta funcion debe llamarse despues de cada problem.fitness(...), tanto si
        la evaluacion fue directa como si el candidato paso el filtro subrogado.
        """
        self._x_reales.append(np.asarray(x, dtype=float).copy())
        self._y_reales.append(float(fitness))
        self.estadisticas.registrar_evaluacion_real()
        
    def puede_usar_subrogado(self):
        """
        Comprueba si ya existe informacion suficiente para activar el filtro.
        """
        if self.config.probabilidad_subrogado <= 0.0:
            return False
        if self.evals_reales < self.warmup_evals:
            return False

        if self._evals_ultimo_reinicio is not None and self.config.cooldown_reinicio_evals > 0:
            evals_desde_reinicio = self.evals_reales - self._evals_ultimo_reinicio
            
            if evals_desde_reinicio < self.config.cooldown_reinicio_evals:
                return False

        return True

    def decidir_evaluacion(self, candidato, fitness_ref, generacion=None):
        """
        Decide si un candidato debe evaluarse realmente.

        La metaheuristica debe llamar a este metodo antes de gastar una
        evaluacion real en un candidato que pueda filtrarse.
        """
        self.estadisticas.registrar_candidato_generado()

        if not self.puede_usar_subrogado():
            return ResultadoFiltroSubrogado(
                debe_evaluar=True,
                uso_subrogado=False,
                decision=None,
                motivo="subrogado_no_activo",
            )

        if self.rng.random() >= self.config.probabilidad_subrogado:
            return ResultadoFiltroSubrogado(
                debe_evaluar=True,
                uso_subrogado=False,
                decision=None,
                motivo="evaluacion_directa_por_probabilidad",
            )

        self.estadisticas.registrar_candidato_con_subrogado()
        self._asegurar_modelo_entrenado()

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
        pred = decision.fitness_pred
        ref = decision.fitness_ref
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
        Registra una evaluacion real que no paso por subrogado.

        Este metodo es util para que las metaheuristicas distingan en las
        estadisticas entre evaluaciones directas y evaluaciones permitidas por
        el filtro.
        """
        self._x_reales.append(np.asarray(x, dtype=float).copy())
        self._y_reales.append(float(fitness))
        self.estadisticas.registrar_evaluacion_directa()

    def registrar_evaluacion_tras_subrogado(self, x, fitness):
        """
        Registra una evaluacion real realizada tras pasar el filtro subrogado.
        """
        self._x_reales.append(np.asarray(x, dtype=float).copy())
        self._y_reales.append(float(fitness))
        self.estadisticas.registrar_evaluacion_tras_subrogado()
        
    def registrar_reinicio(self) -> None:
        """
        Registra un reinicio de la metaheuristica.

        No se borra el historico. La estrategia no acumulativa se aplica
        seleccionando siempre la ventana reciente en el entrenamiento.
        """
        self.estadisticas.registrar_reinicio()
        self.estadisticas.registrar_evento(
            "reinicio",
            evals_reales=int(self.evals_reales),
        )
        
        self._modelo = None
        self._modelo_entrenado_con_n = 0
        self._evals_ultimo_reinicio = int(self.evals_reales)

    def _asegurar_modelo_entrenado(self):
        """
        Entrena el modelo si no existe o si han entrado suficientes evaluaciones
        reales desde el ultimo entrenamiento.
        """
        if self._modelo is not None:
            evals_desde_entrenamiento = self.evals_reales - self._modelo_entrenado_con_n
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
        """
        Devuelve la ventana no acumulativa usada para entrenar el subrogado.
        """
        n = min(self.window_size, self.evals_reales)
        x_train = np.asarray(self._x_reales[-n:], dtype=float)
        y_train = np.asarray(self._y_reales[-n:], dtype=float)
        return x_train, y_train

    def resumen(self):
        """
        Devuelve las estadisticas agregadas junto con la configuracion usada.
        """
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
