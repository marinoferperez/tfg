"""
Adaptación de AGE (estacionario) al benchmark continuo CEC2017 (cec2017real).

CEC2017 está definido como minimización y esta adaptación mantiene
ese criterio en el algoritmo.
"""

from typing import Optional

import numpy as np

from metaheuristics.age.genetic_stationary import GeneticAlgorithm
from metaheuristics.problems.cec2017_problem import CEC2017RealProblem


class GeneticStationaryCEC2017:
    """
    Envoltorio AGE + CEC2017 para ejecutar una función concreta.
    """

    def __init__(self, **age_kwargs) -> None:
        self.age = GeneticAlgorithm(**age_kwargs)

    def optimize(
        self,
        funcid: int,
        dim: int,
        seed: int = 42,
        algname: str = "age_stationary",
        lib_path: Optional[str] = None,
    ):
        seed = int(seed)
        self.age.rng = np.random.default_rng(seed)

        problem = CEC2017RealProblem(
            funcid=funcid,
            dim=dim,
            algname=algname,
            lib_path=lib_path,
            seed=seed,
        )
        problem.prepare_run()

        best_solution, best_cec_fitness = self.age.optimize(
            limites=problem.get_bounds(),
            problem=problem,
        )

        best_error = problem.cec_error(best_cec_fitness)

        return {
            "best_solution": best_solution,
            "best_cec_fitness": float(best_cec_fitness),
            "best_cec_error": best_error,
        }
