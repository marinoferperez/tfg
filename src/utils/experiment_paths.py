"""
Lógica de dominio para la estructura de directorios del benchmark.

Agrupa las funciones que conocen la convención de carpetas del proyecto
(funciones CEC, algoritmos, benchmark_surrogates, metricas_runs, etc.).
Depende de src.utils.fs_utils para resolver y validar rutas.
"""

from pathlib import Path

from src.utils.fs_utils import resolver_ruta, resolver_directorio_existente


ALGORITMOS_MH = ("age", "de", "shade")


def gestiona_algoritmos(valores):
    """Normaliza --algorithm: acepta listas por espacios, comas o 'all'."""
    tokens = []
    for valor in valores or ["all"]:
        for token in str(valor).split(","):
            token = token.strip().lower()
            if token:
                tokens.append(token)
    if not tokens or "all" in tokens:
        return list(ALGORITMOS_MH)
    permitidos = set(ALGORITMOS_MH)
    resultado = []
    for token in tokens:
        if token not in permitidos:
            raise ValueError(
                f"Algoritmo no soportado: {token!r}. Usa age, de, shade o all."
            )
        if token not in resultado:
            resultado.append(token)
    return resultado


# ---------------------------------------------------------------------------
# Normalización de identificadores de función CEC
# ---------------------------------------------------------------------------

def normalizar_funcion(funcion: str | int | None) -> str | None:
    """
    Normaliza un identificador de función CEC al formato 'fN'.

    funcion: entero o cadena tipo '3', 'f3' o 'F3'. None devuelve None.
    """
    if funcion is None:
        return None
    txt = str(funcion).strip().lower()
    if not txt:
        return None
    if txt.startswith("f"):
        txt = txt[1:]
    if not txt.isdigit():
        raise ValueError("La funcion debe ser un entero o un identificador tipo f1.")
    return f"f{int(txt)}"


def clave_funcion(funcion: str | int) -> tuple[int, int | str]:
    """
    Clave de ordenación para identificadores de función CEC.

    funcion: identificador normalizable por normalizar_funcion.
    """
    funcion_norm = normalizar_funcion(funcion)
    if funcion_norm is None:
        raise ValueError("No se puede ordenar una funcion vacia.")
    try:
        return (0, int(funcion_norm[1:]))
    except ValueError:
        return (1, funcion_norm)


# ---------------------------------------------------------------------------
# Inferencia de rutas a partir de la estructura de carpetas
# ---------------------------------------------------------------------------

def buscar_ancestro_nombrado(ruta: str | Path, nombre: str) -> Path | None:
    """
    Busca el primer ancestro (o la propia ruta) con el nombre indicado.

    ruta: punto de partida de la búsqueda.
    nombre: nombre de directorio a encontrar.
    """
    path = resolver_ruta(ruta)
    for parent in (path, *path.parents):
        if parent.name == nombre:
            return parent
    return None


def inferir_benchmark_dir(ruta: str | Path) -> Path | None:
    """
    Infiere el directorio benchmark_surrogates a partir de cualquier ruta interior.

    ruta: ruta dentro de la estructura benchmark_surrogates.
    """
    return buscar_ancestro_nombrado(ruta, "benchmark_surrogates")


def inferir_benchmark_dir_desde_candidatos(*rutas: str | Path | None) -> Path | None:
    """
    Devuelve el primer benchmark_surrogates inferible de la lista de rutas.

    rutas: rutas candidatas (se aceptan None para simplificar el llamador).
    """
    for ruta in rutas:
        if ruta is None:
            continue
        benchmark_dir = inferir_benchmark_dir(ruta)
        if benchmark_dir is not None:
            return benchmark_dir
    return None


def inferir_algoritmo_desde_artefacto(ruta: str | Path) -> str | None:
    """
    Infiere el algoritmo (age/de/shade) a partir de la ruta de un artefacto.

    ruta: ruta a un fichero dentro de la estructura de resultados.
    """
    path = resolver_ruta(ruta)
    for parent in (path, *path.parents):
        if parent.name.lower() in ALGORITMOS_MH:
            return parent.name.lower()
    return None


def inferir_directorio_modelo(dataset_path: str | Path, model_name: str) -> Path | None:
    """
    Infiere el directorio de salida del modelo a partir de la ruta del dataset.

    dataset_path: ruta al dataset preprocesado.
    model_name: nombre del modelo subrogado.
    """
    dataset = resolver_ruta(dataset_path)
    benchmark_dir = inferir_benchmark_dir(dataset)
    if benchmark_dir is None:
        return None

    partes_rel = dataset.relative_to(benchmark_dir).parts
    if len(partes_rel) >= 3 and partes_rel[0] == "preprocesado":
        algoritmo = partes_rel[1]
        model_dir = benchmark_dir / algoritmo / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    if len(partes_rel) >= 2:
        model_dir = dataset.parent / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    return None


def detectar_funciones_experimento(
    experiment_dir: str | Path,
    *,
    funciones: list[str] | None = None,
    required_subdir: str = "benchmark_surrogates",
) -> list[str]:
    """
    Detecta las funciones CEC disponibles en un directorio de experimento.

    experiment_dir: raíz del experimento.
    funciones: lista de funciones a filtrar; None devuelve todas las detectadas.
    required_subdir: subdirectorio que debe existir para considerar la función válida.
    """
    base = resolver_directorio_existente(experiment_dir, arg_name="experiment_dir")
    if funciones:
        normalizadas = sorted(
            {normalizar_funcion(f) for f in funciones}, key=clave_funcion
        )
        faltantes = [
            funcion
            for funcion in normalizadas
            if not (base / funcion / required_subdir).is_dir()
        ]
        if faltantes:
            raise FileNotFoundError(
                f"No se encontraron {required_subdir} para: {', '.join(faltantes)}"
            )
        return normalizadas

    detectadas = [
        path.name
        for path in sorted(base.glob("f*"), key=lambda p: clave_funcion(p.name))
        if (path / required_subdir).is_dir()
    ]
    return detectadas


def resolver_inputs_experimento(
    experiment_dir: str | Path,
    algoritmo: str,
    funcion: str | int | None = None,
) -> list[Path]:
    """
    Resuelve los paths de datasets de entrada para un algoritmo y función CEC.

    experiment_dir: raíz del experimento.
    algoritmo: nombre del algoritmo.
    funcion: identificador de función CEC; None busca en todas.
    """
    base = resolver_directorio_existente(experiment_dir, arg_name="experiment_dir")
    funcion_norm = normalizar_funcion(funcion)
    rutas: list[Path] = []
    base_funciones = (
        base / "metaheuristica_resultados"
        if (base / "metaheuristica_resultados").is_dir()
        else base
    )

    def _extend_datasets(pattern_base: str):
        """Añade a rutas los datasets que coinciden con pattern_base en base."""
        for extension in ("h5", "hdf5"):
            rutas.extend(sorted(base.glob(f"{pattern_base}.{extension}")))

    def _extend_datasets_base_funciones(pattern_base: str):
        """Añade a rutas los datasets que coinciden con pattern_base en base_funciones."""
        for extension in ("h5", "hdf5"):
            rutas.extend(sorted(base_funciones.glob(f"{pattern_base}.{extension}")))

    if (base / "metricas_runs" / "cec2017").exists():
        _extend_datasets(
            f"metricas_runs/cec2017/{algoritmo}/*/dataset_{algoritmo}_cec2017_*"
        )
    else:
        if funcion_norm is None:
            _extend_datasets_base_funciones(
                f"f*/metricas_runs/cec2017/{algoritmo}/*/dataset_{algoritmo}_cec2017_*"
            )
        else:
            _extend_datasets_base_funciones(
                f"{funcion_norm}/metricas_runs/cec2017/{algoritmo}/*/dataset_{algoritmo}_cec2017_*"
            )

    if not rutas:
        raise FileNotFoundError(
            f"No se encontraron dataset_*.h5/.hdf5 para algoritmo={algoritmo} dentro de {base}"
        )
    return rutas

