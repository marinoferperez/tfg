"""
Adaptacion de DE al problema combinatorio QAP.

QAP se modela en minimizacion.
"""

import numpy as np

from metaheuristics.de.differential_evolution import DifferentialEvolution
from metaheuristics.problems.qap_problem import QAPProblem

class DifferentialEvolutionQAP:
    def __init__(self, **de_kwargs):
        self.de = DifferentialEvolution(**de_kwargs)

    def optimize(self, qap_path=None, flow_matrix=None, distance_matrix=None, seed=42):
        seed = int(seed)
        self.de.seed = seed

        if qap_path is not None:
            problema = QAPProblem.from_qaplib(path=qap_path, seed=seed, tipo_algoritmo="continuo")
        else:
            problema = QAPProblem(mat_flujo=flow_matrix, mat_distancias=distance_matrix, seed=seed, tipo_algoritmo="continuo")

        mejor_sol, mejor_fitness = self.de.optimize(limites=problema.get_bounds(), problem=problema)
        mejor_permutacion = problema.decodificar_asignacion(np.asarray(mejor_sol, dtype=float))

        resultado = {
            "mejor_sol": mejor_sol,
            "mejor_fitness": float(mejor_fitness),
            "mejor_permutacion": mejor_permutacion.astype(int).tolist(),
        }
        return resultado
