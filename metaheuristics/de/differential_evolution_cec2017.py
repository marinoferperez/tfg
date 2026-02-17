"""
Adaptación de DE al benchmark continuo CEC2017 (cec2017real).

CEC2017 es un problema de minimización, y DE se ejecuta en ese mismo criterio.
"""

from __future__ import annotations

from typing import Optional

from metaheuristics.de.differential_evolution import DifferentialEvolution
from metaheuristics.problems.cec2017_problem import CEC2017RealProblem


class DifferentialEvolutionCEC2017:
    """
    Envoltorio DE + CEC2017 para ejecutar una función concreta.
    """

    def __init__(self, **de_kwargs) -> None:
        self.de = DifferentialEvolution(**de_kwargs)

    def optimize(
        self,
        funcid: int,
        dim: int,
        seed: int = 42,
        algname: str = "de_cec",
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

        best_solution, best_cec_fitness = self.de.optimize(
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
    optimizer = DifferentialEvolutionCEC2017(
        tam_poblacion=50,
        f=0.5,
        cr=0.9,
        max_evals=2000,
        seed=42,
    )
    result = optimizer.optimize(funcid=1, dim=10, seed=42, algname="de_cec_demo")
    print(f"F01 D10 -> error={result['best_cec_error']:.6e}, cec_fitness={result['best_cec_fitness']:.6e}")
