from pathlib import Path
import re
import numpy as np

class QAPProblem:
    # constructor del problema qap
    # ----------------------------
    # valida matrices y configura la representacion segun tipo_algoritmo
    def __init__(self, mat_flujo, mat_distancias, seed = 42, tipo_algoritmo = "continuo"):
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
        self.dim = self.n
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.flujo = flujo
        self.distance = dist

        self.tipo_algoritmo = str(tipo_algoritmo).strip().lower()
        if self.tipo_algoritmo != "continuo" and self.tipo_algoritmo != "combinatorio":
            raise ValueError("tipo_algoritmo debe ser 'continuo' o 'combinatorio'")


        self.limites = None
        if self.tipo_algoritmo == "continuo":
            # cualquier rango de valores sirve -> nos importa el orden
            self.limites = np.tile(np.array([[0.0, 1.0]], dtype=float), (self.n, 1))

    # from_qaplib carga una instancia qaplib (.dat) y construye el problema
    @classmethod
    def from_qaplib(cls, path, seed=42, tipo_algoritmo="continuo"):
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

        return cls(
            mat_flujo=flujo,
            mat_distancias=dist,
            seed=seed,
            tipo_algoritmo=tipo_algoritmo,
        )

    # get_limites devuelve limites del espacio de busqueda
    # para continuo devuelve [0,1]^n y para combinatorio devuelve None
    def get_limites(self):
        return self.limites

    # alias retrocompatible con problemas continuos que exponen get_bounds()
    def get_bounds(self):
        return self.get_limites()

    # get_size devuelve n (dimension del problema)
    def get_size(self):
        return self.dim

    # create_population genera individuos iniciales segun la representacion elegida
    # def create_population(self, rng=None, pop_size=50, ind_size=None, limites=None):
    #     if rng is None:
    #         rng = self.rng
    #     if ind_size is None:
    #         ind_size = self.n
    #     if int(ind_size) != self.n:
    #         raise ValueError("ind_size debe coincidir con n")

    #     if self.tipo_algoritmo == "continuo":
    #         return rng.uniform(0.0, 1.0, size=(int(pop_size), int(ind_size)))
    #     return np.asarray([rng.permutation(self.n) for _ in range(int(pop_size))], dtype=int)

    # decodificar_asignacion transforma la representacion interna (individuo) en permutacion
    # en continuo aplica argsort y en combinatorio valida directamente
    def decodificar_asignacion(self, solution):
        if self.tipo_algoritmo == "combinatorio":
            arr = np.asarray(solution)
            return self._normaliza_permutacion(arr)

        arr = np.asarray(solution, dtype=float)
        if arr.ndim != 1 or arr.shape[0] != self.n:
            raise ValueError("La solucion debe tener forma (n,)")
        return np.argsort(arr, kind="mergesort")

    # _normaliza_permutacion comprueba que la permutacion sea valida
    # sin truncados silenciosos de flotantes no enteros -> limpia
    def _normaliza_permutacion(self, permutation):
        arr = np.asarray(permutation)

        if arr.ndim != 1 or arr.shape[0] != self.n:
            raise ValueError("La permutacion debe tener forma (n,)")

        if np.issubdtype(arr.dtype, np.floating):
            if not np.all(np.isfinite(arr)):
                raise ValueError("La permutacion no es valida")
            if not np.all(arr == np.floor(arr)):
                raise ValueError("La permutacion debe contener solo enteros")
        elif not np.issubdtype(arr.dtype, np.integer):
            raise ValueError("La permutacion debe contener solo enteros")

        perm = arr.astype(int, copy=False)
        if np.unique(perm).shape[0] != self.n or np.min(perm) < 0 or np.max(perm) >= self.n:
            raise ValueError("La permutacion no es valida")
        return perm

    # evaluar_permutacion calcula el coste qap para una permutacion valida
    def evaluar_permutacion(self, permutation):
        perm = self._normaliza_permutacion(permutation)

        # Costo QAP: sum_i sum_j flujo[i,j] * distance[p(i), p(j)]
        dist_perm = self.distance[np.ix_(perm, perm)]
        return float(np.sum(self.flujo * dist_perm))

    # fitness devuelve el coste para un individuo o para una poblacion
    # respetando la representacion asociada al tipo de algoritmo
    def fitness(self, solution):
        arr = np.asarray(solution)
        if arr.ndim == 1:
            if self.tipo_algoritmo == "continuo":
                perm = self.decodificar_asignacion(arr)
                return self.evaluar_permutacion(perm)
            return self.evaluar_permutacion(arr)
        if arr.ndim == 2:
            if self.tipo_algoritmo == "continuo":
                return np.asarray([self.fitness(ind) for ind in arr], dtype=float)
            return np.asarray([self.evaluar_permutacion(ind) for ind in arr], dtype=float)
        raise ValueError("solution debe tener forma (n,) o (m, n)")
