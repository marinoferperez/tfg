"""
Adaptación de AGE (estacionario) al benchmark continuo CEC2017 (cec2017real).

CEC2017 está definido como minimización y esta adaptación mantiene
ese criterio en el algoritmo.
"""

from __future__ import annotations

from typing import Optional

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
        output_to_console: bool = False,
    ):
        problem = CEC2017RealProblem(
            funcid=funcid,
            dim=dim,
            algname=algname,
            lib_path=lib_path,
            seed=seed,
        )
        problem.prepare_run(output_to_console=output_to_console)

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


if __name__ == "__main__":
    # Ejemplo de humo rápido (puedes subir max_evals para el benchmark oficial).
    optimizer = GeneticStationaryCEC2017(
        tam_poblacion=50,
        prob_cruce=0.7,
        prob_mutacion=0.1,
        sigma=0.3,
        alpha=0.45,
        tam_torneo=2,
        max_evals=2000,
        seed=42,
    )
    result = optimizer.optimize(funcid=1, dim=10, seed=42, algname="age_stationary_demo")
    print(f"F01 D10 -> error={result['best_cec_error']:.6e}, cec_fitness={result['best_cec_fitness']:.6e}")
