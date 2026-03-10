# implementación del algoritmo DE utilizando la librería PYADE.

import numpy as np

# PYADE cuenta con la implementación directa, formal y genérica (básica) del 
# algoritmo DE, en concreto, DE/rand/1 (un solo vector diferencial)/bin (cruce binario)

import pyade.de

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

    def __init__(self, tam_poblacion = None, f = None, cr = None, metodo_cruce = None, max_evals = None, seed = 42):
        self.tam_poblacion = tam_poblacion
        self.f = f
        self.cr = cr
        self.metodo_cruce = metodo_cruce
        self.max_evals = max_evals
        self.seed = seed

        self.evals = 0 # contabilizador de evals realizadas para evitar que evals > max_evals
        self._max_evals_reales = None
        self._problem = None

        self._dataset = None
        self._perm_decodificador = None
        self._generacion_actual = 0

    # evalua_solucion incrementa el contador de evaluaciones cada vez que se ejecuta
    # la funcion objetivo (fitness) y devuelve el fitness (float)

    def evalua_solucion(self, solution):
        # pyade puede superar las max_evals ya que trabaja internamente con max_iters
        # max_iters = max_evals / tam_poblacion 
        # por tanto, y como queremos que el tope sea max_evals, evaluamos aquells individuos
        # que no superen las evaluacioes. para el resto, se devuelve el peor fitness posible evitando
        # que sean escogidos  
        if self._max_evals_reales is not None and self.evals >= self._max_evals_reales:
            return float("inf")

        fit = float(self._problem.fitness(solution))
        self.evals += 1
        
        if self._dataset is not None:
            perm = None
            if self._perm_decodificador is not None:
                perm = self._perm_decodificador(solution)
            
            self._dataset.individuo_to_dataset(
                eval_id = int(self.evals),
                generacion = int(getattr(self, "_generacion_actual", 0)),
                x = np.asarray(solution, dtype = float),
                fitness = float(fit),
                perm = perm
            )
        
        return fit

        # return float(self._problem.fitness(solution))

    # optimize ejecuta DE sobre el problema concreto
    #   * limites : array (dim, 2) con [min, max] por dim
    #   * problem : problema con método fitness
    # devuelve mejor_solucion y mejor_fitness

    def optimize(self, limites, problem, callback_metricas = None, dataset = None, perm_decodificador = None):
        dim = limites.shape[0]
        self.evals = 0
        self._max_evals_reales = None
        self._problem = problem

        ####

        self._dataset = dataset 
        self._perm_decodificador = perm_decodificador

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

        params = pyade.de.get_default_params(dim = dim)

        # se asignan los valores correctos a los parametros principales del algoritmo
        params['individual_size'] = dim
        params['bounds'] = np.asarray(limites, dtype=float)
        params['seed'] = int(self.seed)

        # se sobreescriben los parametros principales si son distintos al default 
        if self.tam_poblacion is not None: params['population_size'] = int(self.tam_poblacion)
        if self.max_evals is not None: params['max_evals'] = int(self.max_evals)
        if self.f is not None: params['f'] = float(self.f)
        if self.cr is not None: params['cr'] = float(self.cr)
        if self.metodo_cruce is not None: params['cross'] = self.metodo_cruce

        if int(params['max_evals']) <= 0:
            raise ValueError("max_evals debe ser > 0")
        self._max_evals_reales = int(params['max_evals'])

        # se asigna la funcion que llama al fitness del problema
        params['func'] = self.evalua_solucion
        # callback opcional para registrar la evolucion por generacion
        if callback_metricas is not None:
            params['callback'] = callback_metricas

        mejor_solucion, mejor_fitness = pyade.de.apply(**params)
        return mejor_solucion, float(mejor_fitness)
