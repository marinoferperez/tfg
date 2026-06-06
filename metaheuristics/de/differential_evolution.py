# implementación del algoritmo DE utilizando la librería PYADE.
#
# Diferencias respecto a la versión anterior (general):
#   - Se elimina el soporte para permutaciones: sin perm_decodificador en
#     optimize() ni lógica perm en evalua_solucion.
#   - El dataset de surrogate ya no serializa 'generacion'; esa información se
#     usa solo internamente para resumir la diversidad por generación en HDF5.

# PYADE cuenta con la implementación directa, formal y genérica (básica) del
# algoritmo DE, en concreto, DE/rand/1 (un solo vector diferencial)/bin (cruce binario)

import numpy as np

from metaheuristics.metrics.reinicio_elitista import (
    ControlReinicioElitista,
    calcular_diversidad_normalizada,
    seleccionar_indice_elitista,
)

try:
    import pyade.commons as pyade_commons
    import pyade.de as pyade_de
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "No se pudo importar 'pyade.de'. Este proyecto usa la libreria PYADE de "
        "https://github.com/xKuZz/pyade . Si tienes instalado otro paquete llamado "
        "'pyade', desinstalalo e instala las dependencias con: "
        "python3 -m pip install -r requirements.txt"
    ) from exc


class DifferentialEvolution:
    # constructor del DE
    # --------------------
    # permite construir como objeto el algoritmo, indicando los valores propios de cada uno de los
    # parámetros esenciales de los que depende PYADE.

    # por defecto, pyade asocia valores a cada uno de sus parametros definidos:
    # * tam_poblacion = 10 * dim
    # * individual_size = 10
    # * max_evals = 10000 * dim
    # * f = 0.5
    # * cr = 0.9
    # * cross = "bin"
    # * seed = None (no asigna por defecto)
    # * callback = None

    # en este caso, fijamos la semilla por defecto --> seed = 42

    def __init__(self, tam_poblacion=None, f=None, cr=None, metodo_cruce=None,
                 max_evals=None, seed=42, reinicio_elitista=False,
                 reinicio_elitista_ratio_estabilidad_diversidad=None,
                 reinicio_elitista_ratio_paciencia=0.05,
                 reinicio_elitista_ventana_evaluaciones=2500):
        self.tam_poblacion = tam_poblacion
        self.f = f
        self.cr = cr
        self.metodo_cruce = metodo_cruce
        self.max_evals = max_evals
        self.seed = seed
        self.reinicio_elitista = bool(
            reinicio_elitista or reinicio_elitista_ratio_estabilidad_diversidad is not None
        )
        self.reinicio_elitista_ratio_estabilidad_diversidad = (
            float(reinicio_elitista_ratio_estabilidad_diversidad)
            if reinicio_elitista_ratio_estabilidad_diversidad is not None
            else None
        )
        self.reinicio_elitista_ratio_paciencia = float(reinicio_elitista_ratio_paciencia)
        self.reinicio_elitista_ventana_evaluaciones = int(reinicio_elitista_ventana_evaluaciones)
        self.eventos_reinicio_elitista = []
        self._control_reinicio_elitista = (
            ControlReinicioElitista(
                self.reinicio_elitista_ratio_estabilidad_diversidad,
                max_evals=max_evals,
                ratio_paciencia=self.reinicio_elitista_ratio_paciencia,
                ventana_evaluaciones=self.reinicio_elitista_ventana_evaluaciones,
            )
            if self.reinicio_elitista
            else None
        )

        self.evals = 0  # contabilizador de evals realizadas para evitar que evals > max_evals
        self._max_evals_reales = None
        self._problem = None

        self._dataset = None
        # _generacion_actual es actualizado por CallbackMetricasDE via setattr.
        # Se usa solo durante la ejecucion para poder resumir despues la
        # diversidad por generacion sin serializar esa columna en el dataset.
        self._generacion_actual = 0

        # callback de la poblacion inicial (PYADE no lo dispara automaticamente)
        self._tam_poblacion_inicial = 0
        self._buffer_fitness_inicial = None
        self._callback_inicial = None
        self._params_de = None

    def _diversidad_normalizada(self, poblacion, rango_inf=-100.0, rango_sup=100.0):
        return calcular_diversidad_normalizada(poblacion)

    def aplicar_reinicio_elitista_desde_estado(self, estado, generacion):
        if self._control_reinicio_elitista is None:
            return False

        population = estado.get("population")
        fitness = estado.get("fitness")
        bounds = estado.get("bounds")
        func = estado.get("func")
        if population is None or fitness is None or bounds is None or func is None:
            return False

        population = np.asarray(population)
        fitness = np.asarray(fitness)
        bounds = np.asarray(bounds, dtype=float)
        if population.ndim != 2 or population.shape[0] < 2:
            return False

        if not self._control_reinicio_elitista.debe_reiniciar(
            fitness=fitness,
            poblacion=population,
            evaluaciones=int(self.evals),
            generacion=generacion,
            bounds=bounds,
        ):
            return False
        diagnostico = self._control_reinicio_elitista.diagnostico_reinicio()

        n_nuevos = int(population.shape[0]) - 1
        evals_antes = int(self.evals)
        evals_restantes = int(self._max_evals_reales) - evals_antes
        if n_nuevos <= 0 or evals_restantes < n_nuevos:
            return False

        elite_idx = seleccionar_indice_elitista(fitness)
        elite = np.asarray(population[elite_idx], dtype=float).copy()
        elite_fit = float(fitness[elite_idx])
        mejor_fit = float(elite_fit)

        nuevos = pyade_commons.init_population(
            int(n_nuevos),
            int(bounds.shape[0]),
            np.asarray(bounds, dtype=float),
        )
        self._generacion_actual = int(generacion)
        nuevos_fit = np.asarray([float(func(ind)) for ind in nuevos], dtype=float)
        self._generacion_actual = int(generacion) + 1

        population[0] = elite
        fitness[0] = elite_fit
        population[1:] = nuevos
        fitness[1:] = nuevos_fit
        self._control_reinicio_elitista.registrar_estado_post_reinicio(
            fitness=fitness,
            evaluaciones=int(self.evals),
        )

        evento = dict(diagnostico)
        evento.update(
            {
                "generacion": int(generacion),
                "evaluaciones_antes_reinicio": int(evals_antes),
                "evaluaciones_despues_reinicio": int(self.evals),
                "indice_individuo_preservado": int(elite_idx),
                "fitness_preservado": float(elite_fit),
                "mejor_fitness": float(mejor_fit),
            }
        )
        self.eventos_reinicio_elitista.append(evento)
        return True

    # evalua_solucion incrementa el contador de evaluaciones cada vez que se ejecuta
    # la funcion objetivo (fitness) y devuelve el fitness (float)

    def evalua_solucion(self, solution):
        # pyade puede superar las max_evals ya que trabaja internamente con max_iters
        # max_iters = max_evals / tam_poblacion
        # por tanto, y como queremos que el tope sea max_evals, evaluamos aquellos
        # individuos que no superen las evaluaciones. para el resto, se devuelve el
        # peor fitness posible evitando que sean escogidos
        if self._max_evals_reales is not None and self.evals >= self._max_evals_reales:
            return float("inf")

        fit = float(self._problem.fitness(solution))
        self.evals += 1

        if self._dataset is not None:
            self._dataset.individuo_to_dataset(
                eval_id=int(self.evals),
                generacion=int(getattr(self, "_generacion_actual", 0)),
                x=np.asarray(solution, dtype=float),
                fitness=float(fit),
            )

        # Acumular fitness de la poblacion inicial. PYADE no llama al callback durante
        # la inicializacion, solo al final de cada generacion. Para registrar el estado
        # inicial en el logbook (evaluaciones = population_size) lo hacemos aqui manualmente.
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
                # A partir de aquí, las siguientes evaluaciones reales ya
                # pertenecen a la primera generación efectiva de PYADE.
                self._generacion_actual = 1
                self._buffer_fitness_inicial = None  # disparar solo una vez

        return fit

    # optimize ejecuta DE sobre el problema concreto
    #   * limites : array (dim, 2) con [min, max] por dim
    #   * problem : problema con método fitness
    # devuelve mejor_solucion y mejor_fitness

    def optimize(self, limites, problem, callback_metricas=None, dataset=None):
        dim = limites.shape[0]
        self.evals = 0
        self.eventos_reinicio_elitista = []
        self._max_evals_reales = None
        self._problem = problem
        self._generacion_actual = 0

        self._dataset = dataset

        # "get_default_params(dim)" devuelve un diccionario de parámetros asociados al DE:
        #   * population_size = tamaño de la poblacion/número de individuos
        #   * individual_size = tamaño del problema/número de variables = dim
        #   * max_evals = numero máximo de evaluaciones del fitness
        #   * f = factor de escala que multiplica al vector diferencial
        #       * f bajo = pasos pequeños --> explotacion
        #       * f alto = pasos grandes --> exploracion
        #   * cr = probabilidad de cruce para la combinación de mutante y trial
        #   * cross = tipo de cruce --> binomial
        #   * seed = semilla aleatoria --> ya implementado (no hay que utilizar el rng)
        #   * callbacks = función que llama PYADE para logging

        params = pyade_de.get_default_params(dim=dim)

        # se asignan los valores correctos a los parametros principales del algoritmo
        params['individual_size'] = dim
        params['bounds'] = np.asarray(limites, dtype=float)
        params['seed'] = int(self.seed)

        # se sobreescriben los parametros principales si son distintos al default
        if self.tam_poblacion is not None: params['population_size'] = int(self.tam_poblacion)
        if self.max_evals is not None:     params['max_evals'] = int(self.max_evals)
        if self.f is not None:             params['f'] = float(self.f)
        if self.cr is not None:            params['cr'] = float(self.cr)
        if self.metodo_cruce is not None:  params['cross'] = self.metodo_cruce

        if int(params['max_evals']) <= 0:
            raise ValueError("max_evals debe ser > 0")
        self._max_evals_reales = int(params['max_evals'])
        if self.reinicio_elitista:
            self._control_reinicio_elitista = ControlReinicioElitista(
                self.reinicio_elitista_ratio_estabilidad_diversidad,
                max_evals=self._max_evals_reales,
                ratio_paciencia=self.reinicio_elitista_ratio_paciencia,
                ventana_evaluaciones=self.reinicio_elitista_ventana_evaluaciones,
            )
        else:
            self._control_reinicio_elitista = None

        # preparar el callback de la poblacion inicial
        if callback_metricas is not None:
            self._tam_poblacion_inicial = int(params['population_size'])
            self._buffer_fitness_inicial = []
            self._callback_inicial = callback_metricas
            self._params_de = dict(params)
        else:
            self._tam_poblacion_inicial = 0
            self._buffer_fitness_inicial = None
            self._callback_inicial = None
            self._params_de = None

        # se asigna la funcion que llama al fitness del problema
        params['func'] = self.evalua_solucion
        # callback opcional para registrar la evolucion por generacion. El reinicio
        # de DE depende del estado interno que PYADE expone mediante este callback,
        # por lo que debe mantenerse incluso cuando no se guardan metricas pesadas.
        if callback_metricas is not None:
            params['callback'] = callback_metricas
        elif self.reinicio_elitista:
            def callback_reinicio(**estado):
                if estado.get("current_generation") is None:
                    return
                gen = int(estado.get("current_generation", 0)) + 1
                self._generacion_actual = gen
                self.aplicar_reinicio_elitista_desde_estado(estado=estado, generacion=gen)

            params['callback'] = callback_reinicio

        mejor_solucion, mejor_fitness = pyade_de.apply(**params)
        return mejor_solucion, float(mejor_fitness)
