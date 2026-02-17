"""
Algoritmo de Evolución Diferencial (DE) usando la librería pyade.

Este módulo encapsula la funcionalidad de pyade para seguir la estructura 
de clases del proyecto en minimización.
"""

import numpy as np
import pyade.de


class DifferentialEvolution:
    """
    Algoritmo de Evolución Diferencial (DE).
    
    Wrapper sobre pyade.de.apply que permite minimizar problemas definidos
    con la interfaz DefaultProblem.
    """
    
    def __init__(self, tam_poblacion=None, f=None, cr=None, metodo_cruce=None, 
                 max_evals=None, seed=42):
        """
        Constructor del DE.
        
        :param tam_poblacion: Tamaño población (None = defecto pyade).
        :param f: Factor mutación (None = defecto pyade, typ. 0.5).
        :param cr: Ratio cruce (None = defecto pyade, typ. 0.9).
        :param metodo_cruce: 'bin'/'exp' (None = defecto pyade 'bin').
        :param max_evals: Evaluaciones máximas (None = defecto pyade).
        :param seed: Semilla aleatoria (Default 42).
        """
        self.tam_poblacion = tam_poblacion
        self.f = f
        self.cr = cr
        self.metodo_cruce = metodo_cruce
        self.max_evals = max_evals
        self.seed = seed
        self.evals = 0
        self._max_evals_effective = None
        self._problem = None

    def evalua_solucion(self, solution):
        """
        Método puente para pyade:
        1. Incrementa el contador de evaluaciones.
        2. Llama al problema (minimización).
        3. Devuelve el fitness tal cual.
        """
        # Corte estricto de evaluaciones para respetar presupuesto CEC.
        if self._max_evals_effective is not None and self.evals >= self._max_evals_effective:
            return float("inf")

        self.evals += 1
        return float(self._problem.fitness(solution))

    def optimize(self, limites, problem):
        """
        Ejecuta Differential Evolution sobre el problema dado.
        
        :param limites: Array (dim, 2) con [min, max] por dimensión.
        :param problem: Instancia de problema con método fitness().
        :return: Tupla (mejor_solución, mejor_fitness).
        """
        dim = limites.shape[0]
        self.evals = 0
        self._max_evals_effective = None
        self._problem = problem
        
        # Obtenemos parámetros por defecto de pyade
        params = pyade.de.get_default_params(dim=dim)
        
        # Sobrescribimos SOLO si el usuario ha especificado un valor
        if self.tam_poblacion is not None:
            params['population_size'] = self.tam_poblacion
        if self.max_evals is not None:
            params['max_evals'] = self.max_evals
        if self.f is not None:
            params['f'] = self.f
        if self.cr is not None:
            params['cr'] = self.cr
        if self.metodo_cruce is not None:
            params['cross'] = self.metodo_cruce
            
        params['individual_size'] = dim
        params['bounds'] = np.asarray(limites, dtype=float)
        params['seed'] = self.seed
        self._max_evals_effective = int(params['max_evals']) if params.get('max_evals') is not None else None
        
        # Pasamos el método envolvente de la clase
        params['func'] = self.evalua_solucion
        
        # Ejecutar pyade (minimización)
        mejor_solucion, mejor_fitness = pyade.de.apply(**params)
        return mejor_solucion, float(mejor_fitness)
