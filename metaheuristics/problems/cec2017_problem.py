"""
Problema CEC2017 (cec2017real) para usar con metaheurísticas del proyecto.

CEC2017 está definido como minimización y este problema mantiene
ese criterio sin invertir el signo del fitness.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Optional

import numpy as np

from metaheuristics.age.default_problem import DefaultProblem


class CEC2017RealProblem(DefaultProblem):
    """
    Wrapper de CEC2017 usando la librería compartida de cec2017real.
    """

    _VALID_DIMS = {2, 5, 10, 30, 50, 100}

    def __init__(
        self,
        funcid: int,
        dim: int,
        algname: str = "age_stationary",
        lib_path: Optional[str] = None,
        seed: int = 42,
        opts=None,
    ) -> None:
        super().__init__(dim=dim, seed=seed, opts=opts)
        if not 1 <= int(funcid) <= 30:
            raise ValueError("funcid debe estar en [1, 30]")
        if int(dim) not in self._VALID_DIMS:
            raise ValueError("dim debe ser una de {2, 5, 10, 30, 50, 100}")

        self.funcid = int(funcid)
        self.dim = int(dim)
        self.algname = str(algname)
        # cec2017real usa `char directory[30]` con formato "results_<algname>".
        # Debe caber incluyendo el terminador nulo: max 29 caracteres visibles.
        if len(f"results_{self.algname}") > 29:
            raise ValueError(
                "algname demasiado largo para cec2017real "
                "(len('results_'+algname) debe ser <= 29)"
            )
        self.bounds = np.tile(np.array([[-100.0, 100.0]], dtype=float), (self.dim, 1))

        self._lib_path = Path(lib_path) if lib_path is not None else self._default_lib_path()
        self._cec_code_dir = self._lib_path.parent.parent
        self._lib = self._load_library(self._lib_path)
        self._configure_signatures()
        self._initialized = False

    @staticmethod
    def _default_lib_path() -> Path:
        base = Path(__file__).resolve().parents[2] / "cec2017real" / "code" / "build"
        candidates = [
            base / "libcec17_test_func.dylib",
            base / "libcec17_test_func.so",
            base / "cec17_test_func.dll",
            base / "libcec17_test_func.dll",
        ]
        for cand in candidates:
            if cand.exists():
                return cand
        return candidates[0]

    @staticmethod
    def _load_library(lib_path: Path):
        if not lib_path.exists():
            raise FileNotFoundError(
                f"No existe la librería CEC2017 en '{lib_path}'. "
                "Compila con: cmake -S cec2017real/code -B cec2017real/code/build "
                "-DCMAKE_POLICY_VERSION_MINIMUM=3.5 && cmake --build cec2017real/code/build -j"
            )
        return ctypes.CDLL(str(lib_path))

    def _configure_signatures(self) -> None:
        self._lib.cec17_init.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        self._lib.cec17_init.restype = None
        self._lib.cec17_print_output.argtypes = []
        self._lib.cec17_print_output.restype = None
        self._lib.cec17_error.argtypes = [ctypes.c_double]
        self._lib.cec17_error.restype = ctypes.c_double
        self._lib.cec17_fitness.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self._lib.cec17_fitness.restype = ctypes.c_double

    def prepare_run(self, output_to_console: bool = False) -> None:
        """
        Inicializa la función CEC2017 para una nueva ejecución.
        """
        self._ensure_input_data_available()
        results_dir = Path.cwd() / f"results_{self.algname}"
        results_dir.mkdir(parents=True, exist_ok=True)
        self._lib.cec17_init(self.algname.encode("utf-8"), self.funcid, self.dim)
        if output_to_console:
            self._lib.cec17_print_output()
        self._initialized = True

    def _ensure_input_data_available(self) -> None:
        source = self._cec_code_dir / "input_data"
        if not source.exists():
            raise FileNotFoundError(f"No existe input_data en '{source}'")
        target = Path.cwd() / "input_data"
        if target.exists():
            return
        try:
            target.symlink_to(source, target_is_directory=True)
        except OSError:
            raise RuntimeError(
                "CEC2017 necesita un directorio 'input_data' en el cwd. "
                "Ejecuta desde 'cec2017real/code' o crea un symlink manual:\n"
                f"ln -s '{source}' '{target}'"
            )

    def get_bounds(self):
        return self.bounds

    def cec_raw_fitness(self, solution: np.ndarray) -> float:
        """
        Evalúa fitness original CEC2017 (minimización).
        Devuelve el valor tal cual sale de la librería C.
        """
        if not self._initialized:
            raise RuntimeError("CEC2017 no inicializado. Llama a prepare_run() antes de evaluar.")
        x = np.ascontiguousarray(solution, dtype=np.float64)
        ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        return float(self._lib.cec17_fitness(ptr))

    def fitness(self, solution: np.ndarray):
        """
        Fitness original CEC2017 (minimización).
        """
        arr = np.asarray(solution, dtype=float)
        if arr.ndim == 1:
            return self.cec_raw_fitness(arr)
        if arr.ndim == 2:
            return np.asarray([self.cec_raw_fitness(ind) for ind in arr], dtype=float)
        raise ValueError("solution debe tener forma (dim,) o (n, dim)")

    def cec_error(self, cec_fitness: float) -> float:
        return float(self._lib.cec17_error(float(cec_fitness)))
