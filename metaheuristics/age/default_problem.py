"""
Clase base genérica para problemas de optimización.

Esta clase define la interfaz que debe implementar cualquier problema
(continuo o combinatorio) para ser resuelto con el AGE.

Inspirada en la estructura de HeuristicProblem/ProblemInfluencer de C++.
"""

import numpy as np


class DefaultProblem:
    """
    Clase base para problemas de optimización.
    
    Los problemas específicos (CEC2017, combinatorios, etc.) deben heredar
    de esta clase e implementar los métodos necesarios. Deberían guardar
    'self.dim' en su constructor.
    """
    
    def __init__(self, dim, seed=42, opts=None):
        """
        Constructor base mínimo para problemas genéricos.

        :param dim: Dimensión del problema.
        :param seed: Semilla del RNG interno.
        :param opts: Opciones opcionales para fitness.
        """
        self.dim = int(dim)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.opts = opts

    def fitness(self, solution):
        """
        Evalúa la función de fitness para una solución dada.
        
        :param solution: Solución a evaluar (np.ndarray para continuo, 
                        list/array para combinatorio).
        :return: Valor de fitness (float). Para maximización.
        """
        raise NotImplementedError("Debe implementar el método fitness()")
    
    def create_population(self, rng=None, pop_size=50, ind_size=None, bounds=None):
        """
        Genera una población aleatoria uniforme (estándar en papers) dentro 
        de los límites dados.
        
        Esta función es genérica para problemas continuos. Para problemas
        combinatorios, se debe usar el método create_population() del problema
        específico.

        :param pop_size: Número de individuos.
        :param ind_size: Dimensión de cada individuo (opcional, default=self.dim).
        :param bounds: Array (ind_size, 2) con [min, max] por gen.
        :return: Población (pop_size, ind_size).
        """
        if rng is None:
            rng = self.rng
        if ind_size is None:
            ind_size = self.dim
        if bounds is None:
            bounds = self.get_bounds()

        lower = bounds[:, 0]
        upper = bounds[:, 1]
        return rng.uniform(lower, upper, size=(pop_size, ind_size))
    
    def get_size(self):
        """
        Devuelve la dimensión/tamaño del problema.
        
        :return: Dimensión del problema.
        """
        return self.dim
    
    def get_bounds(self):
        """
        Devuelve los límites del espacio de búsqueda.
        
        Para problemas continuos: np.ndarray de forma (dim, 2) con [min, max].
        Para problemas combinatorios: puede devolver None o límites discretos.
        
        :return: Límites del problema.
        """
        raise NotImplementedError("Debe implementar el método get_bounds()")
