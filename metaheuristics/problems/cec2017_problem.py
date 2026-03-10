import ctypes
import numpy as np
from pathlib import Path

class CEC2017Problem:
    DIMS_VALIDAS = {2, 5, 10, 30, 50, 100}

    def __init__(self, funcid, dim, algname = "age_stationary", lib_path = None, seed = 42):
        self.dim = int(dim)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        if not 1 <= int(funcid) <= 30:
            raise ValueError("funcid debe estar en [1, 30]")
        if int(dim) not in self.DIMS_VALIDAS:
            raise ValueError("dim debe ser una de {2, 5, 10, 30, 50, 100}")

        self.funcid = int(funcid)
        self.dim = int(dim)
        self.algname = str(algname)

        try:
            self._algname_bytes = self.algname.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError("algname debe contener solo caracteres ASCII para cec2017real") from exc
        # cec2017real usa `char directory[30]` con formato "results_<algname>".
        # Se valida en bytes para evitar desbordes con UTF-8 multibyte.
        if len(b"results_" + self._algname_bytes) > 29:
            raise ValueError(
                "algname demasiado largo para cec2017real "
                "(len(bytes('results_'+algname_ascii)) debe ser <= 29)"
            )
        self.bounds = np.tile(np.array([[-100.0, 100.0]], dtype=float), (self.dim, 1))

        self._lib_path = Path(lib_path) if lib_path is not None else self._default_lib_path()
        self._cec_code_dir = self._lib_path.parent.parent
        self._lib = self._load_library(self._lib_path)
        self._configure_signatures()
        self._initialized = False

    @staticmethod
    def _default_lib_path():
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
    def _load_library(lib_path):
        if not lib_path.exists():
            raise FileNotFoundError(
                f"No existe la librería CEC2017 en '{lib_path}'. "
                "Compila con: cmake -S cec2017real/code -B cec2017real/code/build "
                "-DCMAKE_POLICY_VERSION_MINIMUM=3.5 && cmake --build cec2017real/code/build -j"
            )
        return ctypes.CDLL(str(lib_path))

    def _configure_signatures(self):
        self._lib.cec17_init.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        self._lib.cec17_init.restype = None
        self._lib.cec17_print_output.argtypes = []
        self._lib.cec17_print_output.restype = None
        self._lib.cec17_error.argtypes = [ctypes.c_double]
        self._lib.cec17_error.restype = ctypes.c_double
        self._lib.cec17_fitness.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self._lib.cec17_fitness.restype = ctypes.c_double

    def prepare_run(self):
        self._ensure_input_data_available()
        results_dir = Path.cwd() / f"results_{self.algname}"
        results_dir.mkdir(parents=True, exist_ok=True)
        self._lib.cec17_init(self._algname_bytes, self.funcid, self.dim)
        self._initialized = True

    def _ensure_input_data_available(self):
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

    def get_size(self):
        return self.dim

    def create_population(self, rng=None, pop_size=50, ind_size=None, bounds=None):
        if rng is None:
            rng = self.rng
        if ind_size is None:
            ind_size = self.dim
        if bounds is None:
            bounds = self.get_bounds()

        lower = bounds[:, 0]
        upper = bounds[:, 1]
        return rng.uniform(lower, upper, size=(pop_size, ind_size))

    def fitness(self, solution):
        if not self._initialized:
            raise RuntimeError("CEC2017 no inicializado. Llama a prepare_run() antes de evaluar.")
        x = np.asarray(solution, dtype=np.float64)
        x = np.ascontiguousarray(x, dtype=np.float64)
        ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        return float(self._lib.cec17_fitness(ptr))

    def cec_error(self, cec_fitness):
        return float(self._lib.cec17_error(float(cec_fitness)))
