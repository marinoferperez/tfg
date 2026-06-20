"""
Evolución diferencial (DE/rand/1) para optimización continua.

Delega la ejecución en pyade_de, que implementa la variante DE/rand/1 con cruce
binomial o exponencial. Opcionalmente activa el mecanismo de reinicio elitista por estancamiento mediante ControlReinicioElitista y registro de métricas por generación mediante el protocolo de callback de PYADE.
"""

import numpy as np

from src.metaheuristics.metrics.elitist_restart import (
    ControlReinicioElitista,
    seleccionar_indice_elitista,
)

try:
    import pyade.commons as pyade_commons
    import pyade.de as pyade_de
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "No se pudo importar 'pyade.de'. Este proyecto usa la libreria PYADE de "
        "https://github.com/xKuZz/pyade"
    ) from exc


class DifferentialEvolution:
    """
    Evolución diferencial DE/rand/1/bin para optimización continua.

    Envuelve la implementación de PYADE y añade reinicio elitista opcional,
    registro en dataset y control estricto del presupuesto de evaluaciones.
    """

    def __init__(self, tam_poblacion=None, f=None, cr=None, metodo_cruce=None,
                 max_evals=None, reinicio=False, reinicio_ratio=0.05):
        """
        Constructor del algoritmo.

        tam_poblacion: número de individuos de la población. Si es None, PYADE usa 10 * dim.
        f: factor de escala del vector diferencial.
        cr: probabilidad de cruce.
        metodo_cruce: tipo de cruce ('bin' o 'exp').
        max_evals: presupuesto máximo de evaluaciones. Si es None, PYADE usa 10000 * dim.
        reinicio: activa el mecanismo de reinicio elitista.
        reinicio_ratio: fracción de max_evals usada como ventana de estancamiento.
        """
        self.tam_poblacion = tam_poblacion
        self.f = f
        self.cr = cr
        self.metodo_cruce = metodo_cruce
        self.max_evals = max_evals
        self.seed = None
        self.reinicio = bool(reinicio)
        self.reinicio_ratio = float(reinicio_ratio)

        # reinicio
        self.eventos_reinicio = []      # historial de eventos de reinicio (it y motivo)
        self._gestor_reinicio = None    # se inicializa en cada llamada a optimize

        self.evals = 0
        self._max_evals_reales = None
        self._problema = None
        self._dataset = None
        # actualizado por CallbackMetricasDE; sirve para anotar generacion en el dataset
        self._generacion_actual = 0

        # PYADE no dispara el callback durante la inicializacion
        self._tam_poblacion_inicial = 0
        self._buffer_fitness_inicial = None
        self._callback_inicial = None
        self._params_de = None

    def _crear_gestor_reinicio(self, max_evals):
        """Crea el gestor de reinicio elitista o None si reinicio está desactivado."""
        if self.reinicio:
            return ControlReinicioElitista(
                max_evals=max_evals,
                ratio_estancamiento=self.reinicio_ratio,
            )
        return None

    def _aplicar_reinicio(self, estado, generacion):
        """
        Aplica un reinicio elitista sobre el estado interno expuesto por PYADE.

        estado: diccionario PYADE con claves population, fitness, bounds y func.
        generacion: generación actual.

        Retorna True si se aplicó el reinicio.
        """
        if self._gestor_reinicio is None:
            return False

        poblacion = estado.get("population")
        fitness = estado.get("fitness")
        limites = estado.get("bounds")
        evaluar = estado.get("func")
        if poblacion is None or fitness is None or limites is None or evaluar is None:
            return False

        poblacion = np.asarray(poblacion)
        fitness = np.asarray(fitness)
        limites = np.asarray(limites, dtype=float)
        if poblacion.ndim != 2 or poblacion.shape[0] < 2:
            return False

        if not self._gestor_reinicio.debe_reiniciar(
            fitness=fitness,
            evaluaciones=int(self.evals),
            generacion=generacion,
        ):
            return False
        diagnostico = self._gestor_reinicio.diagnostico_reinicio()

        n_nuevos = int(poblacion.shape[0]) - 1
        evals_antes = int(self.evals)
        evals_restantes = int(self._max_evals_reales) - evals_antes
        if n_nuevos <= 0 or evals_restantes < n_nuevos:
            return False

        # se preserva el mejor individuo (élite) en la primera posición
        elite_idx = seleccionar_indice_elitista(fitness)
        elite = np.asarray(poblacion[elite_idx], dtype=float).copy()
        elite_fit = float(fitness[elite_idx])
        mejor_fit = float(elite_fit)

        # el resto de la población se regenera uniformemente y se evalúa
        nuevos = pyade_commons.init_population(
            int(n_nuevos),
            int(limites.shape[0]),
            np.asarray(limites, dtype=float),
        )
        self._generacion_actual = int(generacion)
        nuevos_fit = np.asarray([float(evaluar(ind)) for ind in nuevos], dtype=float)
        self._generacion_actual = int(generacion) + 1

        poblacion[0] = elite
        fitness[0] = elite_fit
        poblacion[1:] = nuevos
        fitness[1:] = nuevos_fit
        self._gestor_reinicio.registrar_estado_post_reinicio(
            fitness=fitness,
            evaluaciones=int(self.evals),
        )

        # se registra el evento de reinicio para trazabilidad
        evento = dict(diagnostico)
        evento.update(
            {
                "generacion": int(generacion),
                "evals_antes_reinicio": int(evals_antes),
                "evals_despues_reinicio": int(self.evals),
                "indice_individuo_preservado": int(elite_idx),
                "fitness_preservado": float(elite_fit),
                "mejor_fitness": float(mejor_fit),
            }
        )
        self.eventos_reinicio.append(evento)
        return True

    def _evaluar_individuo(self, individuo):
        """
        Evalúa un individuo, registra la muestra y respeta el tope de evaluaciones.

        individuo: vector de decisión.

        Retorna el fitness real o inf si ya se agotó el presupuesto.
        """
        # PYADE puede superar max_evals porque trabaja con max_iters = max_evals / tam_poblacion
        if self._max_evals_reales is not None and self.evals >= self._max_evals_reales:
            return float("inf")

        fit = float(self._problema.fitness(individuo))
        self.evals += 1

        if self._dataset is not None:
            self._dataset.registrar_evaluacion(
                eval_id=int(self.evals),
                generacion=int(getattr(self, "_generacion_actual", 0)),
                x=np.asarray(individuo, dtype=float),
                fitness=float(fit),
            )

        # registrar manualmente la poblacion inicial en el logbook
        if self._buffer_fitness_inicial is not None:
            self._buffer_fitness_inicial.append(float(fit))
            if len(self._buffer_fitness_inicial) >= self._tam_poblacion_inicial:
                fitness_ini = np.asarray(self._buffer_fitness_inicial, dtype=float)
                p = self._params_de or {}
                self._callback_inicial(
                    fitness=fitness_ini,
                    generacion=0,
                    population=None,
                    population_size=p.get("population_size"),
                    individual_size=p.get("individual_size"),
                    max_evals=p.get("max_evals"),
                    f=p.get("f"),
                    cr=p.get("cr"),
                    cross=p.get("cross"),
                    seed=p.get("seed"),
                )
                # a partir de aqui las evaluaciones pertenecen a la primera generacion de PYADE
                self._generacion_actual = 1
                self._buffer_fitness_inicial = None

        return fit

    def optimize(self, limites, problema, callback_metricas=None, dataset=None):
        """
        Ejecuta DE sobre el problema concreto.

        limites: array (dim, 2) con [inferior, superior] por variable.
        problema: instancia del problema con método fitness.
        callback_metricas: opcional para registrar métricas por generación.
        dataset: opcional para registrar evaluaciones en el dataset de entrenamiento.

        Retorna (mejor_solucion, mejor_fitness).
        """
        dim = limites.shape[0]
        self.evals = 0
        self.eventos_reinicio = []
        self._max_evals_reales = None
        self._problema = problema
        self._generacion_actual = 0
        self._dataset = dataset

        params = pyade_de.get_default_params(dim=dim)
        params["individual_size"] = dim
        params["bounds"] = np.asarray(limites, dtype=float)
        params["seed"] = int(self.seed)

        if self.tam_poblacion is not None:
            params["population_size"] = int(self.tam_poblacion)
        if self.max_evals is not None:
            params["max_evals"] = int(self.max_evals)
        if self.f is not None:
            params["f"] = float(self.f)
        if self.cr is not None:
            params["cr"] = float(self.cr)
        if self.metodo_cruce is not None:
            params["cross"] = self.metodo_cruce

        if int(params["max_evals"]) <= 0:
            raise ValueError("max_evals debe ser > 0")
        self._max_evals_reales = int(params["max_evals"])
        self._gestor_reinicio = self._crear_gestor_reinicio(self._max_evals_reales)

        if callback_metricas is not None:
            self._tam_poblacion_inicial = int(params["population_size"])
            self._buffer_fitness_inicial = []
            self._callback_inicial = callback_metricas
            self._params_de = dict(params)
        else:
            self._tam_poblacion_inicial = 0
            self._buffer_fitness_inicial = None
            self._callback_inicial = None
            self._params_de = None

        params["func"] = self._evaluar_individuo
        # el reinicio depende del estado que PYADE expone en el callback
        if callback_metricas is not None:
            params["callback"] = callback_metricas
        elif self.reinicio:
            def callback_reinicio(**estado):
                """Callback mínimo que solo gestiona el reinicio cuando no hay recolector externo."""
                if estado.get("current_generation") is None:
                    return
                gen = int(estado.get("current_generation", 0)) + 1
                self._generacion_actual = gen
                self._aplicar_reinicio(estado=estado, generacion=gen)

            params["callback"] = callback_reinicio

        mejor_solucion, mejor_fitness = pyade_de.apply(**params)
        return mejor_solucion, float(mejor_fitness)
