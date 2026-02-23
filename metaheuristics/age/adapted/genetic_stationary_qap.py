"""
Adaptacion de AGE (estacionario) al problema combinatorio QAP.

QAP se modela en minimizacion.
"""

import numpy as np

from metaheuristics.age.genetic_stationary_combinatorial import GeneticAlgorithmCombinatorio
from metaheuristics.problems.qap_problem import QAPProblem

class GeneticStationaryQAP:
    def __init__(self, **age_kwargs):
        self.age = GeneticAlgorithmCombinatorio(**age_kwargs)

    def optimize(self, qap_path=None, flow_matrix=None, distance_matrix=None, seed=42):
        seed = int(seed)
        self.age.rng = np.random.default_rng(seed)

        if qap_path is not None:
            problema = QAPProblem.from_qaplib(path=qap_path, seed=seed, tipo_algoritmo="combinatorio")
        else:
            if flow_matrix is None or distance_matrix is None:
                raise ValueError("Debes pasar qap_path o (flow_matrix, distance_matrix)")

            problema = QAPProblem(
                mat_flujo=flow_matrix,
                mat_distancias=distance_matrix,
                seed=seed,
                tipo_algoritmo="combinatorio",
            )

        mejor_sol, mejor_fitness = self.age.optimize(problem=problema)
        mejor_permutacion = np.asarray(mejor_sol, dtype=int)

        return {
            "mejor_sol": mejor_sol,
            "mejor_fitness": float(mejor_fitness),
            "best_qap_cost": float(mejor_fitness),
            "mejor_permutacion": mejor_permutacion.astype(int).tolist(),
        }
