"""
Problema QAP (Quadratic Assignment Problem) para usar con metaheuristicas.

Se mantiene la formulacion natural de minimizacion del QAP.
Representacion interna para AGE/DE: random keys (vector real),
que se decodifica a permutacion por argsort.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from metaheuristics.age.default_problem import DefaultProblem


class QAPProblem(DefaultProblem):
    """QAP de minimizacion con codificacion random-keys."""

    def __init__(self, mat_flujo, mat_distancias, seed = 42, opts = None):
        flujo = np.asarray(mat_flujo, dtype=float)
        dist = np.asarray(mat_distancias, dtype=float)

        if flujo.ndim != 2 or dist.ndim != 2:
            raise ValueError("mat_flujo y mat_distancias deben ser matrices 2D")
        if flujo.shape[0] != flujo.shape[1] or dist.shape[0] != dist.shape[1]:
            raise ValueError("mat_flujo y mat_distancias deben ser cuadradas")
        if flujo.shape != dist.shape:
            raise ValueError("mat_flujo y mat_distancias deben tener la misma forma")
        if flujo.shape[0] < 2:
            raise ValueError("El tamano del QAP debe ser >= 2")

        self.n = int(flujo.shape[0])
        self.flujo = flujo
        self.distance = dist
        self.bounds = np.tile(np.array([[0.0, 1.0]], dtype=float), (self.n, 1))

        super().__init__(dim=self.n, seed=seed, opts=opts)

    @classmethod
    def from_qaplib(cls, path: str, seed: int = 42, opts=None) -> "QAPProblem":
        """
        Carga una instancia QAPLIB (.dat) en formato numerico clasico:
        n, seguido de n*n valores de flujo y n*n valores de distancia.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"No existe el fichero QAP: '{file_path}'")

        text = file_path.read_text(encoding="utf-8", errors="ignore")
        tokens = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        if not tokens:
            raise ValueError(f"No se encontraron datos numericos en '{file_path}'")

        n = int(float(tokens[0]))
        if n < 2:
            raise ValueError("La dimension n del QAP debe ser >= 2")

        expected = 1 + 2 * n * n
        if len(tokens) < expected:
            raise ValueError(
                f"Faltan datos en '{file_path}': se esperaban al menos {expected} numeros y hay {len(tokens)}"
            )

        values = np.asarray([float(v) for v in tokens[1:expected]], dtype=float)
        flujo = values[: n * n].reshape(n, n)
        dist = values[n * n : 2 * n * n].reshape(n, n)

        return cls(mat_flujo=flujo, mat_distancias=dist, seed=seed, opts=opts)

    def get_bounds(self) -> np.ndarray:
        return self.bounds

    def create_population(self, rng=None, pop_size=50, ind_size=None, bounds=None) -> np.ndarray:
        if rng is None:
            rng = self.rng
        if ind_size is None:
            ind_size = self.n
        if int(ind_size) != self.n:
            raise ValueError("ind_size debe coincidir con n")
        return rng.uniform(0.0, 1.0, size=(int(pop_size), int(ind_size)))

    def decode_assignment(self, solution: np.ndarray) -> np.ndarray:
        """Convierte random-keys a permutacion (instalacion -> localizacion)."""
        arr = np.asarray(solution, dtype=float)
        if arr.ndim != 1 or arr.shape[0] != self.n:
            raise ValueError("La solucion debe tener forma (n,)")
        return np.argsort(arr, kind="mergesort")

    def evaluate_permutation(self, permutation: np.ndarray) -> float:
        perm = np.asarray(permutation, dtype=int)
        if perm.ndim != 1 or perm.shape[0] != self.n:
            raise ValueError("La permutacion debe tener forma (n,)")
        if np.unique(perm).shape[0] != self.n or np.min(perm) < 0 or np.max(perm) >= self.n:
            raise ValueError("La permutacion no es valida")

        # Costo QAP: sum_i sum_j flujo[i,j] * distance[p(i), p(j)]
        dist_perm = self.distance[np.ix_(perm, perm)]
        return float(np.sum(self.flujo * dist_perm))

    def fitness(self, solution: np.ndarray):
        arr = np.asarray(solution, dtype=float)
        if arr.ndim == 1:
            perm = self.decode_assignment(arr)
            return self.evaluate_permutation(perm)
        if arr.ndim == 2:
            return np.asarray([self.fitness(ind) for ind in arr], dtype=float)
        raise ValueError("solution debe tener forma (n,) o (m, n)")
