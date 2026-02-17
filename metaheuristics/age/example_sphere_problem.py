"""
Ejemplo de implementación de un problema continuo usando DefaultProblem.

Este archivo muestra cómo crear un problema específico (Sphere) heredando
de la clase base DefaultProblem y cómo ejecutar el AGE.
"""

import numpy as np
from metaheuristics.age.default_problem import DefaultProblem
from metaheuristics.age.genetic_stationary import GeneticAlgorithm


class SphereProblem(DefaultProblem):
    """
    Problema de la función Sphere para optimización continua.
    
    f(x) = sum(x_i^2)
    
    AGE trabaja en minimización, por lo que usamos fitness = sum(x_i^2).
    El óptimo global está en x = [0, 0, ..., 0] con fitness = 0.
    """
    
    def __init__(self, dim, bounds=None):
        """
        Constructor del problema Sphere.
        
        :param dim: Dimensión del problema.
        :param bounds: Límites del espacio de búsqueda. Si None, usa [-5.12, 5.12]^dim
        """
        self.dim = dim
        if bounds is None:
            # Límites estándar para Sphere
            self.bounds = np.array([[-5.12, 5.12]] * dim)
        else:
            self.bounds = bounds
    
    def fitness(self, solution):
        """
        Evalúa la función Sphere.
        
        :param solution: Vector de solución (np.ndarray).
        :return: Valor de fitness para minimización.
        """
        # Si recibimos una población, calculamos todos los fitness de golpe
        if len(solution.shape) > 1:
            return np.sum(solution ** 2, axis=1)
        # Si recibimos un solo individuo
        return np.sum(solution ** 2)
    
    def create_population(self, rng, pop_size, ind_size=None, bounds=None):
        """
        Genera una población aleatoria uniforme dentro de los límites.
        
        :param rng: Generador de números aleatorios.
        :param pop_size: Número de individuos.
        :param ind_size: Dimensión de cada individuo (opcional).
        :param bounds: Límites (opcional).
        :return: Población (pop_size, dim) como np.ndarray.
        """
        if ind_size is None:
            ind_size = self.dim
        if bounds is None:
            bounds = self.bounds

        lower = bounds[:, 0]
        upper = bounds[:, 1]
        return rng.uniform(lower, upper, size=(pop_size, ind_size))
    
    def get_bounds(self):
        """
        Devuelve los límites del problema.
        
        :return: Array (dim, 2) con [min, max] por dimensión.
        """
        return self.bounds


# ---------------------------------------------------------------------------
#  Ejemplo de uso
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Crear instancia del problema
    dim = 10
    problem = SphereProblem(dim=dim)
    
    # Crear instancia del algoritmo genético con los nuevos nombres en español
    ga = GeneticAlgorithm(
        tam_poblacion=50,
        prob_cruce=0.7,
        prob_mutacion=0.1,
        sigma=0.3,
        alpha=0.45,
        tam_torneo=2,
        max_evals=10000 * dim,
        seed=42
    )
    
    # Ejecutar optimización
    mejor_solucion, mejor_fitness = ga.optimize(
        limites=problem.get_bounds(),
        problem=problem
    )
    
    print(f"Mejor fitness encontrado (MIN): {mejor_fitness:.6f}")
    print(f"Mejor solución: {mejor_solucion}")
