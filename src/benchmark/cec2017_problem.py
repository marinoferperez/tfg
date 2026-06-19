"""
Adaptador para el benchmark CEC2017 escrito en C++.

Expone una interfaz de problema reutilizable por las metaheurísticas offline y
online: límites del dominio, evaluación de fitness y gestión del entorno de
ejecución requerido por cec2017real.
"""

import ctypes
import os
from pathlib import Path

import numpy as np

_PAQUETE_DIR = Path(__file__).resolve().parent
_CEC_BUILD_DIR = _PAQUETE_DIR / "cec2017real" / "code" / "build"
_NOMBRES_LIBRERIA = (
    "libcec17_test_func.dylib",
    "libcec17_test_func.so",
    "cec17_test_func.dll",
    "libcec17_test_func.dll",
)
_MENSAJE_COMPILACION = (
    "Compila con: cmake -S cec2017real/code -B cec2017real/code/build "
    "-DCMAKE_POLICY_VERSION_MINIMUM=3.5 && cmake --build cec2017real/code/build -j"
)

# CONSTANTES AUXILIARES

_DIMENSIONES_VALIDAS = frozenset({2, 5, 10, 30, 50, 100})
_FUNCID_MIN = 1
_FUNCID_MAX = 30
_LIMITE_INF = -100.0
_LIMITE_SUP = 100.0
_PREFIJO_RESULTS = b"results_"
# char directory[30] en cec17.c
_CEC_DIRECTORY_BUF_SIZE = 30


# FUNCIONES AUXILIARES

def _inicializar_libreria(lib_path=None):
    """
    Resuelve la ruta, carga la librería CEC2017 y configura las firmas ctypes.

    lib_path: ruta explícita a la librería. Si es None, se busca en el
    directorio build del paquete probando extensiones compatibles con el SO.

    Retorna (lib_path, code_dir, lib), donde code_dir es cec2017real/code.
    """
    if lib_path is not None:
        lib_path = Path(lib_path)
    else:
        lib_path = _CEC_BUILD_DIR / _NOMBRES_LIBRERIA[0]
        for nombre in _NOMBRES_LIBRERIA:
            candidato = _CEC_BUILD_DIR / nombre
            if candidato.exists():
                lib_path = candidato
                break

    if not lib_path.exists():
        raise FileNotFoundError(
            f"No existe la librería CEC2017 en '{lib_path}'. {_MENSAJE_COMPILACION}"
        )

    lib = ctypes.CDLL(str(lib_path))
    lib.cec17_init.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    lib.cec17_init.restype = None
    lib.cec17_error.argtypes = [ctypes.c_double]
    lib.cec17_error.restype = ctypes.c_double
    lib.cec17_fitness.argtypes = [ctypes.POINTER(ctypes.c_double)]
    lib.cec17_fitness.restype = ctypes.c_double

    code_dir = lib_path.parent.parent
    return lib_path, code_dir, lib


def _validar_funcid(funcid):
    """
    Comprueba que el identificador de función pertenece al rango CEC2017.

    funcid: índice de la función de prueba, debe estar en [1, 30].
    """
    funcid = int(funcid)
    if not _FUNCID_MIN <= funcid <= _FUNCID_MAX:
        raise ValueError(f"funcid debe estar en [{_FUNCID_MIN}, {_FUNCID_MAX}]")
    return funcid


def _validar_dimension(dim):
    """
    Comprueba que la dimensionalidad es una de las admitidas por CEC2017.

    dim: número de variables del problema.
    """
    dim = int(dim)
    if dim not in _DIMENSIONES_VALIDAS:
        raise ValueError(f"dim debe ser una de {sorted(_DIMENSIONES_VALIDAS)}")
    return dim


def _validar_algname(algname):
    """
    Valida el nombre del algoritmo exigido por cec2017real.

    algname: etiqueta ASCII usada para crear results_<algname>.
    """
    algname = str(algname)
    try:
        algname_bytes = algname.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("algname debe contener solo caracteres ASCII para cec2017real") from exc

    # el buffer C solo admite "results_<algname>" con terminador nulo incluido
    if len(_PREFIJO_RESULTS + algname_bytes) >= _CEC_DIRECTORY_BUF_SIZE:
        max_algname = _CEC_DIRECTORY_BUF_SIZE - len(_PREFIJO_RESULTS) - 1
        raise ValueError(
            "algname demasiado largo para cec2017real "
            f"(como máximo {max_algname} caracteres ASCII en algname)"
        )
    return algname, algname_bytes


def _asegurar_input_data(code_dir):
    """
    Garantiza que input_data/ esté accesible desde el directorio de trabajo.

    code_dir: directorio cec2017real/code que contiene los ficheros de datos
    del benchmark. Si no existe input_data en el directorio, se crea un symlink.
    """
    source = Path(code_dir) / "input_data"
    if not source.exists():
        raise FileNotFoundError(f"No existe input_data en '{source}'")

    target = Path.cwd() / "input_data"
    if target.exists():
        return

    try:
        # la librería C busca los datos de benchmark en ./input_data
        target.symlink_to(source, target_is_directory=True)
    except OSError as exc:
        raise RuntimeError(
            "CEC2017 necesita un directorio 'input_data' en el cwd. "
            "Ejecuta desde 'cec2017real/code' o crea un symlink manual:\n"
            f"ln -s '{source}' '{target}'"
        ) from exc


class CEC2017Problem:
    """
    Problema de optimización continua definido por el benchmark CEC2017.

    Recoge la configuración del experimento, la evaluación de soluciones y
    la preparación del entorno de ejecución para cec2017real.
    """

    DIMENSIONES = _DIMENSIONES_VALIDAS

    def __init__(self, funcid, dim, algname="age", lib_path=None, seed=42, workdir=None):
        """
        Construye una instancia del problema CEC2017.

        funcid: identificador de la función de prueba, en [1, 30].
        dim: dimensionalidad del problema.
        algname: nombre del algoritmo para la salida de cec2017real.
        lib_path: ruta opcional a la librería compilada.
        seed: semilla del generador aleatorio interno.
        workdir: directorio de trabajo para la ejecución. Si es None, se usa el cwd actual.
        """
        self.funcid = _validar_funcid(funcid)
        self.dim = _validar_dimension(dim)
        self.algname, self._algname_bytes = _validar_algname(algname)
        
        # CEC2017 usa el dominio [-100, 100]^dim para todas las funciones
        self.bounds = np.full((self.dim, 2), [_LIMITE_INF, _LIMITE_SUP], dtype=float)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        self._lib_path, self._code_dir, self._lib = _inicializar_libreria(lib_path)

        self._initialized = False
        self._workdir = Path(workdir).resolve() if workdir is not None else Path.cwd().resolve()
        self._prev_cwd = None

    def prepare_run(self):
        """
        Prepara el entorno y deja el benchmark listo para evaluar soluciones.

        Debe llamarse una vez por ejecución, después de enter_workdir() si se
        usa un directorio de trabajo distinto del directorio original.
        """
        _asegurar_input_data(self._code_dir)

        # cec2017real escribe resultados en results_<algname>/
        results_dir = Path.cwd() / f"results_{self.algname}"
        results_dir.mkdir(parents=True, exist_ok=True)

        self._lib.cec17_init(self._algname_bytes, self.funcid, self.dim)
        self._initialized = True

    def enter_workdir(self):
        """
        Cambia el proceso al directorio de trabajo del experimento.

        Guarda el directorio anterior para poder restaurarlo con exit_workdir().
        """
        self._workdir.mkdir(parents=True, exist_ok=True)
        if self._prev_cwd is None:
            self._prev_cwd = Path.cwd().resolve()
        os.chdir(self._workdir)

    def exit_workdir(self):
        """
        Restaura el directorio de trabajo previo a enter_workdir().

        Si no se había entrado en un workdir, no hace nada.
        """
        if self._prev_cwd is None:
            return
        os.chdir(self._prev_cwd)
        self._prev_cwd = None

    def get_bounds(self):
        """
        Devuelve los límites del dominio de búsqueda.

        Retorna un array de forma (dim, 2) con [inferior, superior] por variable.
        """
        return self.bounds

    def get_size(self):
        """
        Devuelve la dimensionalidad del problema.
        """
        return self.dim

    def fitness(self, solution):
        """
        Evalúa el fitness de una solución con la función CEC2017 activa.

        solution: vector de decisión de longitud dim. Requiere haber llamado antes a prepare_run().
        """
        if not self._initialized:
            raise RuntimeError("CEC2017 no inicializado. Llama a prepare_run() antes de evaluar.")

        x = np.ascontiguousarray(np.asarray(solution, dtype=np.float64))
        ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        return float(self._lib.cec17_fitness(ptr))

    def cec_error(self, cec_fitness):
        """
        Convierte un fitness bruto en el error oficial del benchmark.

        cec_fitness: valor devuelto por fitness().
        """
        return float(self._lib.cec17_error(float(cec_fitness)))
