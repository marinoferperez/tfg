"""Resolucion de rutas de entrada y salida del benchmark de modelos subrogados."""

from pathlib import Path

import numpy as np

from src.utils.fs_utils import resolver_archivo_existente
from src.utils.experiment_paths import normalizar_funcion, resolver_inputs_experimento
from src.utils.dataset_utils import inferir_seed


def ordenar_rutas_por_seed(rutas):
    """
    Ordena una lista de rutas por el numero de semilla inferido de cada fichero.

    rutas: iterable de rutas a ficheros de dataset.
    """
    return sorted((Path(ruta) for ruta in rutas), key=lambda p: inferir_seed(p))


def construir_nombre_bloque(train_pct_ini, train_pct_fin):
    """
    Construye el nombre del subdirectorio para un bloque de entrenamiento.

    train_pct_ini: porcentaje de inicio del bloque.
    train_pct_fin: porcentaje de fin del bloque.
    """
    return f"{int(train_pct_ini)}-{int(train_pct_fin)}"


def seleccionar_datasets(args):
    """
    Selecciona la lista de datasets de entrada para el benchmark.

    args: namespace de argparse con inputs, experiment_dir, algorithm, cec_funcid, max_seeds y selection_seed.
    """
    if args.inputs:
        rutas = [resolver_archivo_existente(path, arg_name="inputs") for path in args.inputs]
    else:
        rutas = resolver_inputs_experimento(
            args.experiment_dir,
            args.algorithm,
            funcion=args.cec_funcid,
        )
    rutas = ordenar_rutas_por_seed(rutas)
    
    if args.max_seeds is not None:
        if args.max_seeds < 1:
            raise ValueError("max_seeds debe ser >= 1.")
        if args.max_seeds > len(rutas):
            raise ValueError(
                f"max_seeds={args.max_seeds} excede el numero de datasets disponibles ({len(rutas)})."
            )
        rng = np.random.default_rng(int(args.selection_seed))
        idx = rng.choice(len(rutas), size=int(args.max_seeds), replace=False)
        rutas = ordenar_rutas_por_seed(rutas[i] for i in idx)
        
    if not rutas:
        raise ValueError("No hay datasets de entrada para ejecutar el benchmark.")
    return rutas


def inferir_raiz_experimento(dataset_path, funcion):
    """
    Infiere la raiz del experimento a partir de la ruta de un dataset.

    dataset_path: ruta a un fichero de dataset de la ejecucion.
    funcion: nombre o id de la funcion CEC (p.ej. "f1" o 1).
    """
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
        return funcion_dir.parent.parent
    
    return funcion_dir.parent


def rutas_salida_benchmark(args, dataset_paths, protocolo):
    """
    Calcula las rutas de salida del benchmark para un modelo y algoritmo dados.

    args: namespace de argparse con output_dir, benchmark_subdir, algorithm, model, out, runs_out, runs_json_out.
    dataset_paths: lista de rutas a los datasets de entrada.
    protocolo: protocolo de evaluacion (p.ej. "no_acumulativo").
    """
    funcion_norm = normalizar_funcion(args.cec_funcid)
    raiz_experimento = inferir_raiz_experimento(dataset_paths[0], args.cec_funcid)
    raiz_salida = Path(args.output_dir).resolve() if getattr(args, "output_dir", None) else raiz_experimento
    benchmark_dir = raiz_salida / Path(args.benchmark_subdir) / protocolo / funcion_norm
    model_dir = benchmark_dir / args.algorithm / args.model

    ruta_metricas = Path(args.out).resolve() if args.out else model_dir / f"{args.model}_metricas.json"
    ruta_runs_csv = Path(args.runs_out).resolve() if args.runs_out else model_dir / f"{args.model}_runs.csv"
    
    if args.runs_json_out:
        ruta_runs_json = Path(args.runs_json_out).resolve()
    elif getattr(args, "save_runs_json", False):
        ruta_runs_json = model_dir / f"{args.model}_runs.json"
    else:
        ruta_runs_json = None
    return benchmark_dir, model_dir, ruta_metricas, ruta_runs_csv, ruta_runs_json


def rutas_bloque_modelo(model_dir, model_name, train_pct_ini, train_pct_fin):
    """
    Calcula las rutas de salida para un bloque especifico dentro del directorio del modelo.

    model_dir: directorio raiz del modelo dentro del benchmark.
    model_name: nombre del modelo subrogado.
    train_pct_ini: porcentaje de inicio del bloque de entrenamiento.
    train_pct_fin: porcentaje de fin del bloque de entrenamiento.
    """
    bloque_dir = Path(model_dir) / construir_nombre_bloque(train_pct_ini, train_pct_fin)
    ruta_metricas = bloque_dir / f"{model_name}_metricas.json"
    ruta_runs_csv = bloque_dir / f"{model_name}_runs.csv"
    
    return bloque_dir, ruta_metricas, ruta_runs_csv
