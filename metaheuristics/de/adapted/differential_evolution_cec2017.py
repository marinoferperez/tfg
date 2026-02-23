"""
Adaptación de DE al benchmark continuo CEC2017 (cec2017real).

CEC2017 es un problema de minimización, y DE se ejecuta en ese mismo criterio.
"""

from metaheuristics.de.differential_evolution import DifferentialEvolution
from metaheuristics.problems.cec2017_problem import CEC2017Problem

class DifferentialEvolutionCEC2017:
    def __init__(self, **de_kwargs):
        self.de = DifferentialEvolution(**de_kwargs)

    def optimize(self, funcid, dim, seed=42, algname="de_cec", lib_path=None):
        seed = int(seed)
        self.de.seed = seed

        problema = CEC2017Problem(funcid=funcid, dim=dim, algname=algname, lib_path=lib_path, seed=seed)
        problema.prepare_run()

        mejor_sol, mejor_fitness = self.de.optimize(
            limites=problema.get_bounds(),
            problem=problema,
        )

        mejor_error = problema.cec_error(mejor_fitness)

        return {
            "mejor_sol": mejor_sol,
            "mejor_fitness": float(mejor_fitness),
            "mejor_error": mejor_error,
        }
