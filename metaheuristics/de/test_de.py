"""
Ejemplo de uso de Differential Evolution con SphereProblem.
"""

import numpy as np
from metaheuristics.age.example_sphere_problem import SphereProblem
from metaheuristics.de.differential_evolution import DifferentialEvolution


if __name__ == "__main__":
    # 1. Definir el problema (Esfera en 10 dimensiones)
    dim = 10
    problem = SphereProblem(dim=dim)
    limites = problem.get_bounds()
    
    # 2. Configurar DE
    # Usamos parámetros típicos: F=0.5, CR=0.9
    de = DifferentialEvolution(
        tam_poblacion=50,
        f=0.5,
        cr=0.9,
        max_evals=10000 * dim,
        seed=42
    )
    
    # 3. Optimizar
    mejor_solucion, mejor_fitness = de.optimize(limites, problem)
    
    print("--- Resultados DE ---")
    print(f"Mejor fitness encontrado: {mejor_fitness:.6f}")
    print(f"Evaluaciones realizadas: {de.evals}")
    # El óptimo de sum(x^2) es 0 (minimización)
    assert mejor_fitness < 1e-5
    print("✓ Test DE PASSED")
