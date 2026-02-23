"""
Adaptacion de AGE (estacionario) al problema combinatorio QAP.

QAP se modela en minimizacion.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from metaheuristics.age.genetic_stationary import GeneticAlgorithm
from metaheuristics.problems.qap_problem import QAPProblem


class GeneticStationaryQAP:
    """Envoltorio AGE + QAP con interfaz de alto nivel."""

    def __init__(self, **age_kwargs) -> None:
        self.age = GeneticAlgorithm(**age_kwargs)

    def optimize(
        self,
        qap_path: Optional[str] = None,
        flow_matrix: Optional[Sequence[Sequence[float]]] = None,
        distance_matrix: Optional[Sequence[Sequence[float]]] = None,
        seed: int = 42,
    ):
        seed = int(seed)
        self.age.rng = np.random.default_rng(seed)

        if qap_path is not None:
            problem = QAPProblem.from_qaplib(path=qap_path, seed=seed)
        else:
            if flow_matrix is None or distance_matrix is None:
                raise ValueError("Debes pasar qap_path o (flow_matrix, distance_matrix)")
            problem = QAPProblem(
                flow_matrix=flow_matrix,
                distance_matrix=distance_matrix,
                seed=seed,
            )

        best_solution, best_fitness = self.age.optimize(
            limites=problem.get_bounds(),
            problem=problem,
        )

        best_assignment = problem.decode_assignment(np.asarray(best_solution, dtype=float))

        result = {
            "best_solution": best_solution,
            "best_fitness": float(best_fitness),
            "best_qap_cost": float(best_fitness),
            "best_assignment": best_assignment.astype(int).tolist(),
        }
        return result
