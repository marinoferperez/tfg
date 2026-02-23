"""
Adaptación de AGE (estacionario) al benchmark continuo CEC2017 (cec2017real).

CEC2017 está definido como minimización y esta adaptación mantiene
ese criterio en el algoritmo.
"""

import numpy as np

from metaheuristics.age.genetic_stationary_continuous import GeneticAlgorithmContinuo
from metaheuristics.problems.cec2017_problem import CEC2017Problem

class GeneticStationaryCEC2017:
    def __init__(self, **age_kwargs):
        self.age = GeneticAlgorithmContinuo(**age_kwargs)

    def optimize(self, funcid, dim, seed=42, algname="age", lib_path=None):
        seed = int(seed)
        self.age.rng = np.random.default_rng(seed)

        problema = CEC2017Problem(funcid=funcid, dim=dim, algname=algname, lib_path=lib_path, seed=seed)
        problema.prepare_run()

        mejor_sol, mejor_fitness = self.age.optimize(
            limites=problema.get_bounds(),
            problem=problema,
        )

        mejor_error = problema.cec_error(mejor_fitness)

        return {
            "mejor_sol": mejor_sol,
            "mejor_fitness": float(mejor_fitness),
            "mejor_error": mejor_error,
        }
