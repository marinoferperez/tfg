from __future__ import annotations

from pathlib import Path

import numpy as np

from preprocesado_de_datos.utils.path_utils import (
    normalizar_funcion,
    resolver_archivo_existente,
    resolver_inputs_experimento,
)
from preprocesado_de_datos.utils.utils import inferir_seed


def ordenar_paths_por_seed(paths):
    return sorted((Path(path) for path in paths), key=lambda p: inferir_seed(p))


def construir_nombre_batch(train_pct_ini, train_pct_fin):
    return f"{int(train_pct_ini)}-{int(train_pct_fin)}"


def resolver_inputs_benchmark(args):
    if args.inputs:
        rutas = [resolver_archivo_existente(path, arg_name="inputs") for path in args.inputs]
    else:
        rutas = resolver_inputs_experimento(
            args.experiment_dir,
            args.algoritmo,
            funcion=args.funcion,
        )
    rutas = ordenar_paths_por_seed(rutas)
    if args.max_seeds is not None:
        if args.max_seeds < 1:
            raise ValueError("max_seeds debe ser >= 1.")
        if args.max_seeds > len(rutas):
            raise ValueError(
                f"max_seeds={args.max_seeds} excede el numero de datasets disponibles ({len(rutas)})."
            )
        rng = np.random.default_rng(int(args.seed_selection_random_state))
        idx = rng.choice(len(rutas), size=int(args.max_seeds), replace=False)
        rutas = ordenar_paths_por_seed(rutas[i] for i in idx)
    if not rutas:
        raise ValueError("No hay datasets de entrada para ejecutar el benchmark.")
    return rutas


def resolver_funcion_y_raiz_experimento(dataset_path, funcion):
    funcion_norm = normalizar_funcion(funcion)
    dataset_path = Path(dataset_path).resolve()
    funcion_dir = None
    for parent in dataset_path.parents:
        try:
            if normalizar_funcion(parent.name) == funcion_norm:
                funcion_dir = parent
                break
        except ValueError:
            continue
    if funcion_dir is None:
        raise ValueError(f"No se pudo inferir la carpeta de {funcion_norm} a partir de {dataset_path}.")
    if funcion_dir.parent.name == "metaheuristica_resultados":
        raiz_experimento = funcion_dir.parent.parent
    else:
        raiz_experimento = funcion_dir.parent
    return funcion_dir, raiz_experimento


def resolver_rutas_salida_benchmark(args, dataset_paths, protocolo):
    funcion_norm = normalizar_funcion(args.funcion)
    _, raiz_experimento = resolver_funcion_y_raiz_experimento(dataset_paths[0], args.funcion)
    benchmark_subdir = Path(args.benchmark_subdir)
    benchmark_dir = raiz_experimento / benchmark_subdir / protocolo / funcion_norm
    model_dir = benchmark_dir / args.algoritmo / args.model

    ruta_metricas = Path(args.out).resolve() if args.out else model_dir / f"{args.model}_metricas.json"
    ruta_runs_csv = Path(args.runs_out).resolve() if args.runs_out else model_dir / f"{args.model}_runs.csv"
    ruta_runs_json = (
        Path(args.runs_json_out).resolve()
        if args.runs_json_out
        else None
        if getattr(args, "no_runs_json", False)
        else model_dir / f"{args.model}_runs.json"
    )
    ruta_errores = Path(args.errors_out).resolve() if args.errors_out else None
    return benchmark_dir, model_dir, ruta_metricas, ruta_runs_csv, ruta_runs_json, ruta_errores


def resolver_rutas_salida_no_acumulativo(args, dataset_paths, protocolo):
    return resolver_rutas_salida_benchmark(args, dataset_paths, protocolo)


def resolver_rutas_batch_modelo(
    model_dir,
    model_name,
    train_pct_ini,
    train_pct_fin,
    *,
    errores_suffix=None,
):
    batch_dir = Path(model_dir) / construir_nombre_batch(train_pct_ini, train_pct_fin)
    ruta_metricas = batch_dir / f"{model_name}_metricas.json"
    ruta_runs_csv = batch_dir / f"{model_name}_runs.csv"
    ruta_runs_json = batch_dir / f"{model_name}_runs.json"
    ruta_errores = None
    if errores_suffix:
        ruta_errores = batch_dir / f"{model_name}_errores{errores_suffix}"
    return batch_dir, ruta_metricas, ruta_runs_csv, ruta_runs_json, ruta_errores
