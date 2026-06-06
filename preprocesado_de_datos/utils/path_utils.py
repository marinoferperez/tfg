from __future__ import annotations

import csv
import json
import math
from numbers import Integral, Real
from pathlib import Path

import pandas as pd


ALGORITMOS_MH = ("age", "de", "shade")
DECIMALES_METRICAS = 4


def normalizar_funcion(funcion: str | int | None) -> str | None:
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
    funcion_norm = normalizar_funcion(funcion)
    if funcion_norm is None:
        raise ValueError("No se puede ordenar una funcion vacia.")
    try:
        return (0, int(funcion_norm[1:]))
    except ValueError:
        return (1, funcion_norm)


def resolver_ruta(ruta: str | Path) -> Path:
    return Path(ruta).expanduser().resolve()


def resolver_archivo_existente(ruta: str | Path, *, arg_name: str = "ruta") -> Path:
    path = resolver_ruta(ruta)
    if not path.is_file():
        raise FileNotFoundError(f"No existe el archivo indicado en {arg_name}: {path}")
    return path


def resolver_directorio_existente(ruta: str | Path, *, arg_name: str = "ruta") -> Path:
    path = resolver_ruta(ruta)
    if not path.is_dir():
        raise FileNotFoundError(f"No existe el directorio indicado en {arg_name}: {path}")
    return path


def asegurar_directorio(ruta: str | Path) -> Path:
    path = Path(ruta)
    path.mkdir(parents=True, exist_ok=True)
    return path


def asegurar_directorio_padre(ruta: str | Path) -> Path:
    path = Path(ruta)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def leer_json(ruta: str | Path):
    path = resolver_archivo_existente(ruta, arg_name="json")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def redondear_metricas(valor, *, decimales: int = DECIMALES_METRICAS):
    if isinstance(valor, dict):
        return {
            clave: redondear_metricas(subvalor, decimales=decimales)
            for clave, subvalor in valor.items()
        }
    if isinstance(valor, list):
        return [redondear_metricas(item, decimales=decimales) for item in valor]
    if isinstance(valor, tuple):
        return tuple(redondear_metricas(item, decimales=decimales) for item in valor)
    if isinstance(valor, bool) or valor is None or isinstance(valor, str):
        return valor
    if isinstance(valor, Integral):
        return int(valor)
    if isinstance(valor, Real):
        numero = float(valor)
        if not math.isfinite(numero):
            return numero
        numero = round(numero, int(decimales))
        if numero.is_integer():
            return int(numero)
        return numero
    return valor


def preparar_filas_csv(filas, *, decimales: int = DECIMALES_METRICAS):
    return [
        {
            campo: redondear_metricas(valor, decimales=decimales)
            for campo, valor in fila.items()
        }
        for fila in filas
    ]


def escribir_json(ruta: str | Path, payload) -> Path:
    path = asegurar_directorio_padre(ruta)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def escribir_csv_dicts(
    ruta: str | Path,
    filas,
    *,
    fieldnames: list[str] | None = None,
) -> Path:
    path = asegurar_directorio_padre(ruta)
    if not filas:
        path.write_text("", encoding="utf-8")
        return path

    campos = fieldnames or list(filas[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=campos, lineterminator="\n")
        writer.writeheader()
        for fila in filas:
            writer.writerow({campo: fila.get(campo) for campo in campos})
    return path


def leer_csv(ruta: str | Path, **kwargs) -> pd.DataFrame:
    path = resolver_archivo_existente(ruta, arg_name="csv")
    return pd.read_csv(path, **kwargs)


def escribir_csv(
    df: pd.DataFrame,
    ruta: str | Path,
    *,
    index: bool = False,
    **kwargs,
) -> Path:
    path = asegurar_directorio_padre(ruta)
    df.to_csv(path, index=index, **kwargs)
    return path


def resolver_input_cli(
    *,
    input_opt: str | Path | None = None,
    input_pos: str | Path | None = None,
    arg_name: str = "input",
) -> Path:
    if input_opt is not None and input_pos is not None:
        raise ValueError(f"Usa solo una forma de indicar {arg_name}: opcion o argumento posicional.")
    raw = input_opt if input_opt is not None else input_pos
    if raw is None:
        raise ValueError(f"Debes indicar {arg_name}.")
    return resolver_archivo_existente(raw, arg_name=arg_name)


def resolver_paths_cli(
    *,
    inputs_opt: list[str] | None = None,
    inputs_pos: list[str] | None = None,
    allow_empty: bool = False,
) -> list[str]:
    inputs_opt = inputs_opt or []
    inputs_pos = inputs_pos or []
    if inputs_opt and inputs_pos:
        raise ValueError("Usa solo una forma de indicar las rutas de entrada: opcion o argumentos posicionales.")
    seleccion = inputs_opt or inputs_pos
    if not seleccion and not allow_empty:
        raise ValueError("Debes indicar al menos una ruta de entrada.")
    return seleccion


def detectar_algoritmos_benchmark(benchmark_dir: str | Path) -> list[str]:
    base = resolver_directorio_existente(benchmark_dir, arg_name="benchmark_dir")
    candidatos = []
    for path in sorted(base.iterdir()):
        if not path.is_dir():
            continue
        if path.name == "preprocesado":
            continue
        if any(path.glob("*/*_metricas.json")):
            candidatos.append(path.name)
    return candidatos


def listar_metricas_json_algoritmo(benchmark_dir: str | Path, algoritmo: str) -> list[Path]:
    base = resolver_directorio_existente(benchmark_dir, arg_name="benchmark_dir")
    algoritmo_dir = base / algoritmo
    if not algoritmo_dir.is_dir():
        return []
    return sorted(algoritmo_dir.glob("*/*_metricas.json"))


def detectar_funciones_experimento(
    experiment_dir: str | Path,
    *,
    funciones: list[str] | None = None,
    required_subdir: str = "benchmark_surrogates",
) -> list[str]:
    base = resolver_directorio_existente(experiment_dir, arg_name="experiment_dir")
    if funciones:
        normalizadas = sorted({normalizar_funcion(f) for f in funciones}, key=clave_funcion)
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


def buscar_ancestro_nombrado(ruta: str | Path, nombre: str) -> Path | None:
    path = resolver_ruta(ruta)
    for parent in (path, *path.parents):
        if parent.name == nombre:
            return parent
    return None


def inferir_benchmark_dir(ruta: str | Path) -> Path | None:
    return buscar_ancestro_nombrado(ruta, "benchmark_surrogates")


def inferir_benchmark_dir_desde_candidatos(*rutas: str | Path | None) -> Path | None:
    for ruta in rutas:
        if ruta is None:
            continue
        benchmark_dir = inferir_benchmark_dir(ruta)
        if benchmark_dir is not None:
            return benchmark_dir
    return None


def _es_metadata_balanceado(path: Path) -> bool:
    return path.name.endswith(".metadata.json") and "dataset_balanceado" in path.name


def resolver_metadatas_balanceado(paths: list[str] | None = None, *, search_root: str | Path | None = None) -> list[Path]:
    if not paths:
        base = resolver_ruta(search_root) if search_root is not None else Path.cwd().resolve()
        return sorted(p.resolve() for p in base.rglob("*.metadata.json") if _es_metadata_balanceado(p))

    encontrados: list[Path] = []
    for raw in paths:
        path = resolver_ruta(raw)
        if path.is_file():
            if not _es_metadata_balanceado(path):
                raise ValueError(
                    f"El archivo indicado no es una metadata de dataset_balanceado: {path}"
                )
            encontrados.append(path)
            continue

        if path.name == "benchmark_surrogates":
            encontrados.extend(sorted(p.resolve() for p in path.rglob("*.metadata.json") if _es_metadata_balanceado(p)))
            continue

        benchmark_dir = path / "benchmark_surrogates"
        if benchmark_dir.is_dir():
            encontrados.extend(
                sorted(p.resolve() for p in benchmark_dir.rglob("*.metadata.json") if _es_metadata_balanceado(p))
            )
            continue

        encontrados.extend(sorted(p.resolve() for p in path.rglob("*.metadata.json") if _es_metadata_balanceado(p)))

    unicos: list[Path] = []
    vistos: set[Path] = set()
    for path in encontrados:
        if path in vistos:
            continue
        vistos.add(path)
        unicos.append(path)
    return unicos


def inferir_algoritmo_desde_artefacto(ruta: str | Path) -> str | None:
    path = resolver_ruta(ruta)
    for parent in (path, *path.parents):
        if parent.name.lower() in ALGORITMOS_MH:
            return parent.name.lower()
    return None


def inferir_paths_runs_originales_desde_benchmark(
    benchmark_dir: str | Path,
    algoritmo: str,
) -> list[Path]:
    base = resolver_directorio_existente(benchmark_dir, arg_name="benchmark_dir")
    metricas_runs = base.parent / "metricas_runs"
    patrones = [
        metricas_runs / "cec2017" / algoritmo,
        metricas_runs / "qap" / algoritmo,
    ]
    rutas: list[Path] = []
    for root in patrones:
        if root.is_dir():
            rutas.extend(sorted(root.rglob("dataset_*.npz")))
    return rutas


def resolver_inputs_experimento(
    experiment_dir: str | Path,
    algoritmo: str,
    funcion: str | int | None = None,
) -> list[Path]:
    base = resolver_directorio_existente(experiment_dir, arg_name="experiment_dir")
    funcion_norm = normalizar_funcion(funcion)
    rutas: list[Path] = []
    base_funciones = base / "metaheuristica_resultados" if (base / "metaheuristica_resultados").is_dir() else base

    def _extend_datasets(pattern_base: str):
        for extension in ("npz", "h5", "hdf5"):
            rutas.extend(sorted(base.glob(f"{pattern_base}.{extension}")))

    def _extend_datasets_base_funciones(pattern_base: str):
        for extension in ("npz", "h5", "hdf5"):
            rutas.extend(sorted(base_funciones.glob(f"{pattern_base}.{extension}")))

    if (base / "metricas_runs" / "qap").exists():
        _extend_datasets(f"metricas_runs/qap/{algoritmo}/*/dataset_{algoritmo}_qap_*")

    if (base / "metricas_runs" / "cec2017").exists():
        _extend_datasets(f"metricas_runs/cec2017/{algoritmo}/*/dataset_{algoritmo}_cec2017_*")
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
            f"No se encontraron dataset_*.npz/.h5/.hdf5 para algoritmo={algoritmo} dentro de {base}"
        )
    return rutas


def detectar_tareas_seleccion_bins(
    experiment_dir: str | Path,
    *,
    algoritmo: str = "todos",
    funcion: str | int | None = None,
) -> list[dict[str, str | None]]:
    base = resolver_directorio_existente(experiment_dir, arg_name="experiment_dir")
    algoritmos = list(ALGORITMOS_MH) if algoritmo == "todos" else [algoritmo]
    funcion_norm = normalizar_funcion(funcion)
    tareas: list[dict[str, str | None]] = []
    base_funciones = base / "metaheuristica_resultados" if (base / "metaheuristica_resultados").is_dir() else base

    if (base / "metricas_runs" / "cec2017").exists():
        nombre_funcion = normalizar_funcion(base.name)
        if nombre_funcion is None:
            raise ValueError(
                f"No se pudo inferir la funcion CEC a partir de {base}. Usa una carpeta fX o la raiz del experimento."
            )
        raiz_experimento = base.parent.parent if base.parent.name == "metaheuristica_resultados" else base.parent
        for alg in algoritmos:
            tareas.append(
                {
                    "algoritmo": alg,
                    "funcion": nombre_funcion,
                    "experiment_dir": str(base),
                    "out": str(
                        raiz_experimento
                        / "benchmarking"
                        / "offline"
                        / "dataset_preprocesado"
                        / "seleccion_muestras"
                        / nombre_funcion
                        / f"{alg}_seleccion_bins.json"
                    ),
                }
            )
        return tareas

    funciones = sorted(
        (p.name for p in base_funciones.iterdir() if p.is_dir() and p.name.lower().startswith("f")),
        key=clave_funcion,
    )
    if funciones:
        if funcion_norm is not None:
            funciones = [f for f in funciones if f == funcion_norm]
        for fun in funciones:
            fun_dir = base_funciones / fun
            if not (fun_dir / "metricas_runs" / "cec2017").exists():
                continue
            for alg in algoritmos:
                tareas.append(
                    {
                        "algoritmo": alg,
                        "funcion": fun,
                        "experiment_dir": str(fun_dir),
                        "out": str(
                            base
                            / "benchmarking"
                            / "offline"
                            / "dataset_preprocesado"
                            / "seleccion_muestras"
                            / fun
                            / f"{alg}_seleccion_bins.json"
                        ),
                    }
                )
        if tareas:
            return tareas

    if (base / "metricas_runs" / "qap").exists():
        for alg in algoritmos:
            tareas.append(
                {
                    "algoritmo": alg,
                    "funcion": None,
                    "experiment_dir": str(base),
                    "out": str(
                        base
                        / "benchmarking"
                        / "offline"
                        / "dataset_preprocesado"
                        / "seleccion_muestras"
                        / f"{alg}_seleccion_bins.json"
                    ),
                }
            )
        return tareas

    raise FileNotFoundError(f"No se reconocio una estructura valida de experimento en {base}.")


def detectar_tareas_dataset_completo(
    experiment_dir: str | Path,
    *,
    algoritmo: str = "todos",
    funcion: str | int | None = None,
) -> list[dict[str, str | None]]:
    base = resolver_directorio_existente(experiment_dir, arg_name="experiment_dir")
    algoritmos = list(ALGORITMOS_MH) if algoritmo == "todos" else [algoritmo]
    funcion_norm = normalizar_funcion(funcion)
    tareas: list[dict[str, str | None]] = []
    base_funciones = base / "metaheuristica_resultados" if (base / "metaheuristica_resultados").is_dir() else base

    if (base / "metricas_runs" / "cec2017").exists():
        nombre_funcion = normalizar_funcion(base.name)
        if nombre_funcion is None:
            raise ValueError(
                f"No se pudo inferir la funcion CEC a partir de {base}. Usa una carpeta fX o la raiz del experimento."
            )
        raiz_experimento = base.parent.parent if base.parent.name == "metaheuristica_resultados" else base.parent
        outdir_funcion = (
            raiz_experimento / "metaheuristica_resultados" / nombre_funcion
            if (raiz_experimento / "metaheuristica_resultados").is_dir()
            else base
        )
        for alg in algoritmos:
            tareas.append(
                {
                    "algoritmo": alg,
                    "funcion": nombre_funcion,
                    "experiment_dir": str(base),
                    "out": str(outdir_funcion / "dataset_completo" / alg / "dataset_completo.npz"),
                }
            )
        return tareas

    funciones = sorted(
        (p.name for p in base_funciones.iterdir() if p.is_dir() and p.name.lower().startswith("f")),
        key=clave_funcion,
    )
    if funciones:
        if funcion_norm is not None:
            funciones = [f for f in funciones if f == funcion_norm]
        for fun in funciones:
            fun_dir = base_funciones / fun
            if not (fun_dir / "metricas_runs" / "cec2017").exists():
                continue
            for alg in algoritmos:
                tareas.append(
                    {
                        "algoritmo": alg,
                        "funcion": fun,
                        "experiment_dir": str(fun_dir),
                        "out": str(
                            fun_dir
                            / "dataset_completo"
                            / alg
                            / "dataset_completo.npz"
                        ),
                    }
                )
        if tareas:
            return tareas

    if (base / "metricas_runs" / "qap").exists():
        outdir_qap = (base / "metaheuristica_resultados") if (base / "metaheuristica_resultados").is_dir() else base
        for alg in algoritmos:
            tareas.append(
                {
                    "algoritmo": alg,
                    "funcion": None,
                    "experiment_dir": str(base),
                    "out": str(outdir_qap / "dataset_completo" / alg / "dataset_completo.npz"),
                }
            )
        return tareas

    raise FileNotFoundError(f"No se reconocio una estructura valida de experimento en {base}.")


def inferir_directorio_modelo(dataset_path: str | Path, model_name: str) -> Path | None:
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
