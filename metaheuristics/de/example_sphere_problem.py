"""
Ejemplo básico de uso de DifferentialEvolution con un problema continuo.
"""

import numpy as np

from metaheuristics.age.default_problem import DefaultProblem
from metaheuristics.de.differential_evolution import DifferentialEvolution


class SphereProblem(DefaultProblem):
    """
    Sphere en minimización:
    fitness(x) = sum(x_i^2)
    """

    def __init__(self, dim, bounds=None, **kwargs):
        super().__init__(dim, **kwargs)
        if bounds is None:
            self.bounds = np.array([[-5.12, 5.12]] * dim)
        else:
            self.bounds = np.asarray(bounds, dtype=float)

    def fitness(self, solution):
        return np.sum(np.asarray(solution, dtype=float) ** 2)

    def get_bounds(self):
        return self.bounds


if __name__ == "__main__":
    dim = 10
    problem = SphereProblem(dim=dim)

    de = DifferentialEvolution(
        tam_poblacion=10 * dim,
        f=0.5,
        cr=0.5,
        max_evals=10000 * dim,
        seed=42,
    )

    best_solution, best_fitness = de.optimize(limites=problem.get_bounds(), problem=problem)

    print(f"Mejor fitness encontrado: {best_fitness:.6f}")
    print(f"Mejor solución: {best_solution}")
