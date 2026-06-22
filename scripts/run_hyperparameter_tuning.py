"""
Benchmark offline de modelos subrogados con ajuste interno de hiperparametros.

Implementa la estrategia temporal no acumulativa por bloques del 20%: para cada
bloque de entrenamiento se elige la mejor configuracion de hiperparametros mediante
validacion interna y se evalua sobre el bloque siguiente. Compatible con RBF, SVR,
MLP, RSM, Random Forest, HGB, Lasso y XGBoost.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.surrogates.preprocessing.scaling import (
    MODELOS_ARBOL,
    ajustar_y,
    construir_escalador_y,
    escalar_X,
    invertir_y,
)
from src.utils.fs_utils import resolver_archivo_existente
from src.utils.experiment_paths import gestiona_algoritmos, normalizar_funcion
from src.utils.experiment_io import mostrar
from src.utils.dataset_utils import cargar_dataset, inferir_seed
from src.utils.benchmark.blocks_eval_splitter import (
    N_BLOQUES,
    TOL_MEJORA_BLOQUE_ABS,
    TOL_MEJORA_BLOQUE_REL,
    VAL_RATIO_TRAIN,
    construir_casos_acumulativos,
    construir_casos_no_acumulativos,
    truncar_por_convergencia,
)
from src.utils.benchmark.benchmark_io import (
    guardar_artefactos_bloques,
    guardar_artefactos_modelo,
    imprimir_resumen,
    resumir_runs,
)
from src.utils.benchmark.surrogate_paths import (
    seleccionar_datasets,
    rutas_salida_benchmark,
)
from src.surrogates.evaluation.metrics import (
    calcular_metricas,
    METRICAS_MAXIMIZAR,
    METRICAS_MINIMIZAR,
)
from src.surrogates.select_model import select_model, MODELOS


STRATEGIES = {
    "cumulative": {
        "protocol": "acumulativo",
        "split_strategy": "temporal_acumulativo_futuro_tuned",
        "constructor_casos": construir_casos_acumulativos,
    },
    "non_cumulative": {
        "protocol": "no_acumulativo",
        "split_strategy": "temporal_no_acumulativo_futuro_tuned",
        "constructor_casos": construir_casos_no_acumulativos,
    },
}


def expandir_funciones(funcion_arg):
    """Convierte el argumento --cec-funcid (lista nargs="+") en una tupla de nombres normalizados."""
    tokens = []
    for parte in funcion_arg:
        for tk in str(parte).split(","):
            tk = tk.strip().lower()
            if tk:
                tokens.append(tk)
    if "all" in tokens:
        return tuple(f"f{i}" for i in range(1, 31))
    return tuple(normalizar_funcion(tk) for tk in tokens)


def expandir_modelos(model_arg):
    """Convierte el argumento --model en una tupla de nombres de modelo validados."""
    txt = str(model_arg).strip().lower()
    if txt == "all":
        return MODELOS
    partes = [parte.strip() for parte in txt.split(",") if parte.strip()]
    modelos = partes if len(partes) > 1 else [txt]
    invalidos = [modelo for modelo in modelos if modelo not in MODELOS]
    if invalidos:
        raise ValueError(f"Modelos no soportados: {', '.join(invalidos)}")
    return tuple(modelos)


def cargar_param_grid(model_name, ruta_json):
    """Carga el grid de hiperparametros desde JSON o construye el grid por defecto del modelo."""
    if ruta_json is not None:
        ruta = resolver_archivo_existente(ruta_json, arg_name="param_grid_json")
        with ruta.open("r", encoding="utf-8") as fh:
            raw_grid = json.load(fh)
        if isinstance(raw_grid, list):
            return [dict(item) for item in raw_grid]
        if isinstance(raw_grid, dict):
            keys = list(raw_grid.keys())
            values = [raw_grid[k] if isinstance(raw_grid[k], list) else [raw_grid[k]] for k in keys]
            return [dict(zip(keys, combo)) for combo in itertools.product(*values)]
        raise ValueError("El grid de parametros debe ser una lista de diccionarios o un diccionario de listas.")

    if model_name == "rsm":
        return [{"degree": 1}, {"degree": 2}, {"degree": 3}]
    if model_name == "rbf":
        grid = []
        for neighbors, smoothing in itertools.product((25, 50, 100), (1e-3, 1e-2)):
            grid.append({"neighbors": neighbors, "smoothing": smoothing, "kernel": "linear", "degree": -1})
        for neighbors, smoothing in itertools.product((50, 100), (1e-3, 1e-2)):
            grid.append({"neighbors": neighbors, "smoothing": smoothing, "kernel": "gaussian", "degree": -1, "epsilon": 1.0})
        return grid
    if model_name in MODELOS:
        return [{}]
    raise ValueError(f"Modelo no soportado: {model_name!r}.")


def build_parser():
    """Construye y retorna el parser de argumentos del benchmark con ajuste."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark temporal no acumulativo con ajuste interno de hiperparametros "
            "dentro de cada bloque de entrenamiento."
        )
    )
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGIES),
        required=True,
        help="Estrategia de evaluacion temporal: cumulative | non_cumulative.",
    )
    parser.add_argument(
        "--experiment-dir",
        required=True,
        help="Directorio raiz del experimento con los datasets por seed.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directorio raiz donde guardar los resultados del ajuste. "
            "Si no se indica, se genera automaticamente como results/tuning_<nombre_experimento>."
        ),
    )
    parser.add_argument(
        "--benchmark-subdir",
        default="tuning_benchmark",
        help=(
            "Subdirectorio relativo dentro del experimento donde guardar los resultados "
            "del benchmark con ajuste. Por defecto: tuning_benchmark."
        ),
    )
    parser.add_argument(
        "--algorithm",
        nargs="+",
        default=["all"],
        help="Metaheuristica a ejecutar. Acepta age, de, shade, all, listas separadas por espacios o comas. Por defecto all.",
    )
    parser.add_argument(
        "--cec-funcid",
        nargs="+",
        required=True,
        help="Funciones CEC2017 a evaluar. Acepta lista (1 4 10), CSV (1,4,10) o 'all'.",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help="Lista opcional de datasets por seed. Si no se indica, se resuelven desde --experiment-dir.",
    )
    parser.add_argument(
        "--model",
        default="all",
        help="Modelo subrogado a evaluar. Acepta nombre unico, lista CSV o 'all'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para reproducibilidad del modelo. Por defecto 42.",
    )
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=None,
        help="Numero maximo de seeds a usar. Si no se indica, se usan todas las disponibles.",
    )
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=42,
        help=(
            "Semilla para seleccionar aleatoriamente el subconjunto de seeds cuando "
            "se usa --max-seeds. Por defecto 42."
        ),
    )
    parser.add_argument(
        "--convergence-truncation",
        action="store_true",
        default=False,
        help=(
            "Truncar el dataset en el ultimo bloque con mejora real del running best. "
            "Los bloques posteriores a la ultima mejora se descartan antes de construir los casos."
        ),
    )
    parser.add_argument(
        "--param-grid-json",
        default=None,
        help="JSON con el grid de hiperparametros a explorar. Si no se indica, se usa el grid por defecto del modelo.",
    )
    parser.add_argument(
        "--tuning-metric",
        default="spearman",
        choices=sorted(METRICAS_MAXIMIZAR | METRICAS_MINIMIZAR),
        help="Metrica usada para seleccionar los mejores hiperparametros en la validacion interna. Por defecto spearman.",
    )
    parser.add_argument(
        "--inner-validation-ratio",
        type=float,
        default=0.20,
        help="Fraccion final del bloque de entrenamiento usada como validacion interna. Por defecto 0.20.",
    )
    parser.add_argument(
        "--store-tuning-results",
        action="store_true",
        help="Guardar el detalle completo de todos los candidatos del grid en cada run.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Ruta de salida para el JSON de metricas agregadas. Por defecto se genera automaticamente.",
    )
    parser.add_argument(
        "--runs-out",
        default=None,
        help="Ruta de salida para el CSV de metricas por run. Por defecto se genera automaticamente.",
    )
    parser.add_argument(
        "--runs-json-out",
        default=None,
        help="Ruta de salida para el JSON de metricas por run. Por defecto se genera automaticamente.",
    )
    parser.add_argument(
        "--save-runs-json",
        action="store_true",
        help="Generar tambien *_runs.json. Por defecto solo se guarda *_runs.csv.",
    )
    parser.add_argument(
        "--bloque-runs",
        action="store_true",
        help=(
            "Generar tambien *_runs.csv por bloque. Por defecto solo se guardan "
            "metricas por bloque para poder construir resumenes."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Si se indica, muestra informacion de progreso por terminal. Por defecto False.",
    )
    return parser


def dividir_train_validacion_interna(train_idx, ratio):
    """Parte train_idx en sub-train e inner-val segun el ratio indicado."""
    train_idx = np.asarray(train_idx, dtype=np.int64)
    if not 0.0 < float(ratio) < 0.5:
        raise ValueError("--inner-validation-ratio debe estar entre 0 y 0.5.")
    n_val = int(np.floor(train_idx.size * float(ratio)))
    if n_val < 1 or train_idx.size - n_val < 1:
        raise ValueError("No hay suficientes muestras para validacion interna.")
    return train_idx[:-n_val], train_idx[-n_val:]


def ajustar_y_predecir(model_name, params, x_train, y_train, x_val):
    """Entrena el modelo con params sobre x_train/y_train y predice sobre x_val."""
    t0 = time.perf_counter()
    y_scaler = construir_escalador_y(model_name)
    y_train_fit = ajustar_y(y_scaler, y_train)

    model = select_model(model_name, **params)
    model.fit(x_train, y_train_fit)
    tiempo_entrenamiento = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred = model.predict(x_val)
    tiempo_prediccion = time.perf_counter() - t1

    y_pred = invertir_y(y_scaler, y_pred)
    return np.asarray(y_pred, dtype=float).ravel(), float(tiempo_entrenamiento), float(tiempo_prediccion)


def es_mejor(score, best_score, metric):
    """Retorna True si score mejora best_score segun la direccion de la metrica."""
    if not np.isfinite(score):
        return False
    if best_score is None:
        return True
    if metric in METRICAS_MAXIMIZAR:
        return score > best_score
    return score < best_score


def seleccionar_parametros(
    *,
    model_name,
    param_grid,
    x,
    y,
    train_idx,
    metric,
    inner_validation_ratio,
):
    """
    Busca los mejores hiperparametros del modelo sobre la validacion interna.

    Itera el param_grid, ajusta sobre inner-train, evalua sobre inner-val y
    retorna (best_params, best_score, resultados_detallados).
    """
    inner_train_idx, inner_val_idx = dividir_train_validacion_interna(train_idx, inner_validation_ratio)
    x_inner_train = x[inner_train_idx]
    y_inner_train = y[inner_train_idx]
    x_inner_val = x[inner_val_idx]
    y_inner_val = y[inner_val_idx]

    best_params = None
    best_score = None
    resultados = []
    for params in param_grid:
        params = dict(params)
        try:
            y_pred, tiempo_entrenamiento, tiempo_prediccion = ajustar_y_predecir(
                model_name,
                params,
                x_inner_train,
                y_inner_train,
                x_inner_val,
            )
            metricas = calcular_metricas(y_inner_val, y_pred)
            score = float(metricas[metric])
            error = None
        except Exception as exc:  # noqa: BLE001 - se registra el fallo y se prueba el siguiente candidato.
            score = float("nan")
            metricas = {}
            tiempo_entrenamiento = float("nan")
            tiempo_prediccion = float("nan")
            error = f"{type(exc).__name__}: {exc}"

        resultados.append(
            {
                "params": params,
                "score": score,
                "metric": metric,
                "error": error,
                "metricas": metricas,
                "train_time_s": tiempo_entrenamiento,
                "predict_time_s": tiempo_prediccion,
            }
        )
        if es_mejor(score, best_score, metric):
            best_score = score
            best_params = params

    if best_params is None:
        raise RuntimeError("Ninguna configuracion de hiperparametros produjo una metrica valida.")
    return best_params, float(best_score), resultados


def ejecutar_benchmark_tuned(
    *,
    dataset_paths,
    funcion,
    algoritmo,
    model_name,
    param_grid,
    tuning_metric,
    inner_validation_ratio,
    random_state,
    seed_selection_random_state,
    truncar_convergencia,
    store_tuning_results,
    constructor_casos,
    protocol,
    split_strategy,
):
    """
    Ejecuta el benchmark con ajuste de hiperparametros sobre los datasets dados.

    Para cada seed y cada bloque de entrenamiento: busca los mejores hiperparametros
    por validacion interna, ajusta el modelo final y evalua sobre el bloque siguiente.
    Retorna el dict de metricas agregadas listo para guardar.
    """
    escalar_y = model_name not in MODELOS_ARBOL
    metricas_runs = []
    seeds_sin_casos = []
    convergencia_por_seed = {}

    for dataset_path in dataset_paths:
        dataset = cargar_dataset(dataset_path)
        seed_dataset = int(inferir_seed(dataset_path))

        convergencia_ultimo_bloque = N_BLOQUES
        convergencia_fraccion = 1.0
        if truncar_convergencia:
            dataset, convergencia_ultimo_bloque, convergencia_fraccion = truncar_por_convergencia(dataset)
        convergencia_por_seed[seed_dataset] = {
            "ultimo_bloque_informativo": int(convergencia_ultimo_bloque),
            "fraccion_retenida": float(convergencia_fraccion),
        }

        x = escalar_X(np.asarray(dataset["x"], dtype=float))
        y = np.asarray(dataset["fitness"], dtype=float).ravel()
        eval_id = np.asarray(dataset["eval_id"], dtype=np.int64)
        casos = constructor_casos(dataset, random_state=random_state)
        if not casos:
            seeds_sin_casos.append(seed_dataset)
            continue

        for caso in casos:
            train_idx = np.asarray(caso["train_idx"], dtype=np.int64)
            val_idx = np.asarray(caso["val_idx"], dtype=np.int64)

            best_params, best_score, tuning_resultados = seleccionar_parametros(
                model_name=model_name,
                param_grid=param_grid,
                x=x,
                y=y,
                train_idx=train_idx,
                metric=tuning_metric,
                inner_validation_ratio=inner_validation_ratio,
            )

            x_train = x[train_idx]
            x_val = x[val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]

            t0 = time.perf_counter()
            y_scaler = construir_escalador_y(model_name)
            y_train_fit = ajustar_y(y_scaler, y_train)

            model = select_model(model_name, **best_params)
            model.fit(x_train, y_train_fit)
            tiempo_entrenamiento = time.perf_counter() - t0

            t1 = time.perf_counter()
            y_pred = model.predict(x_val)
            tiempo_prediccion = time.perf_counter() - t1
            y_pred = invertir_y(y_scaler, y_pred)

            metricas_run = calcular_metricas(y_val, y_pred)
            metricas_run.update(
                {
                    "funcion": str(funcion),
                    "algoritmo": str(algoritmo),
                    "seed": int(caso["seed"]),
                    "bloque_entrenamiento": int(caso["bloque_entrenamiento"]),
                    "train_pct_ini": int(caso["train_pct_ini"]),
                    "train_pct_fin": int(caso["train_pct_fin"]),
                    "etiqueta_bloque": str(caso["etiqueta_bloque"]),
                    "bloques_entrenamiento": ",".join(str(v) for v in caso.get("bloques_entrenamiento", [])),
                    "bloque_validacion": int(caso["bloque_validacion"]),
                    "n_train": int(caso["n_train"]),
                    "n_test": int(caso["n_val"]),
                    "train_idx_hash": str(caso["train_idx_hash"]),
                    "val_idx_hash": str(caso["val_idx_hash"]),
                    "eval_id_train_min": int(np.min(eval_id[train_idx])),
                    "eval_id_train_max": int(np.max(eval_id[train_idx])),
                    "eval_id_val_min": int(np.min(eval_id[val_idx])),
                    "eval_id_val_max": int(np.max(eval_id[val_idx])),
                    "train_time_s": float(tiempo_entrenamiento),
                    "predict_time_s": float(tiempo_prediccion),
                    "convergencia_ultimo_bloque": int(convergencia_ultimo_bloque),
                    "convergencia_fraccion": float(convergencia_fraccion),
                    "convergencia_aplicada": bool(truncar_convergencia),
                    "tuning_metric": str(tuning_metric),
                    "tuning_inner_validation_ratio": float(inner_validation_ratio),
                    "tuning_best_score": float(best_score),
                    "tuning_best_params": best_params,
                    "tuning_n_candidates": int(len(param_grid)),
                }
            )
            if store_tuning_results:
                metricas_run["tuning_resultados"] = tuning_resultados
            metricas_runs.append(metricas_run)

    metricas = resumir_runs(metricas_runs)
    metricas.update(
        {
            "funcion": str(funcion),
            "algoritmo": str(algoritmo),
            "datasets": [str(Path(path).resolve()) for path in dataset_paths],
            "selected_seeds": [int(inferir_seed(path)) for path in dataset_paths],
            "n_datasets_entrada": int(len(dataset_paths)),
            "model": model_name,
            "feature_mode": "x",
            "model_params": {
                "tuning": True,
                "tuning_metric": tuning_metric,
                "inner_validation_ratio": float(inner_validation_ratio),
                "param_grid": param_grid,
            },
            "split_strategy": split_strategy,
            "protocol": protocol,
            "n_bloques": int(N_BLOQUES),
            "validation_ratio": float(VAL_RATIO_TRAIN),
            "scale_features": True,
            "scale_target": bool(escalar_y),
            "escalado": True,
            "random_state": int(random_state),
            "seed_selection_random_state": (
                None if seed_selection_random_state is None else int(seed_selection_random_state)
            ),
            "convergencia_criterio": (
                "mejora_practica_por_bloque" if truncar_convergencia else ""
            ),
            "convergencia_tol_abs": (
                float(TOL_MEJORA_BLOQUE_ABS) if truncar_convergencia else None
            ),
            "convergencia_tol_rel": (
                float(TOL_MEJORA_BLOQUE_REL) if truncar_convergencia else None
            ),
            "seeds_sin_casos_validacion": [int(seed) for seed in seeds_sin_casos],
            "n_seeds_sin_casos_validacion": int(len(seeds_sin_casos)),
            "convergencia_por_seed": convergencia_por_seed,
        }
    )
    return metricas


def main():
    """Punto de entrada del benchmark de ajuste de hiperparametros."""
    parser = build_parser()
    args = parser.parse_args()

    if args.output_dir is None:
        nombre_exp = Path(args.experiment_dir).resolve().name
        args.output_dir = str(ROOT / "results" / f"tuning_{nombre_exp}")

    cfg = STRATEGIES[args.strategy]

    mostrar(args, "Configuracion offline (ajuste):", flush=True)
    mostrar(args, f"  strategy={args.strategy}", flush=True)
    mostrar(args, f"  experiment_dir={args.experiment_dir}", flush=True)
    mostrar(args, f"  output_dir={args.output_dir}", flush=True)
    mostrar(args, f"  algorithm={args.algorithm}", flush=True)
    mostrar(args, f"  cec_funcid={args.cec_funcid}", flush=True)
    mostrar(args, f"  model={args.model}", flush=True)
    mostrar(args, f"  random_state={args.seed}", flush=True)
    mostrar(args, f"  max_seeds={args.max_seeds}", flush=True)
    mostrar(args, f"  convergence_truncation={args.convergence_truncation}", flush=True)
    mostrar(args, f"  tuning_metric={args.tuning_metric}", flush=True)
    mostrar(args, f"  inner_validation_ratio={args.inner_validation_ratio}", flush=True)
    mostrar(args, f"  benchmark_subdir={args.benchmark_subdir}", flush=True)

    funciones = expandir_funciones(args.cec_funcid)
    algoritmos = gestiona_algoritmos(args.algorithm)
    modelos = expandir_modelos(args.model)
    ejecucion_multiple = len(funciones) > 1 or len(algoritmos) > 1 or len(modelos) > 1
    salidas_explicitas = any((args.out, args.runs_out, args.runs_json_out))
    if ejecucion_multiple and args.inputs:
        raise SystemExit("--inputs solo puede utilizarse con una unica funcion y una unica metaheuristica.")
    if ejecucion_multiple and salidas_explicitas:
        raise SystemExit(
            "Las rutas de salida explicitas solo pueden utilizarse con una unica funcion y una unica metaheuristica."
        )

    for funcion in funciones:
        for algoritmo in algoritmos:
            for modelo in modelos:
                args_run = argparse.Namespace(**vars(args))
                args_run.cec_funcid = funcion
                args_run.algorithm = algoritmo
                args_run.model = modelo
                param_grid = cargar_param_grid(args_run.model, args_run.param_grid_json)
                if not param_grid:
                    raise SystemExit("El grid de hiperparametros esta vacio.")
                dataset_paths = seleccionar_datasets(args_run)
                (
                    _,
                    model_dir,
                    ruta_metricas,
                    ruta_runs_csv,
                    ruta_runs_json,
                ) = rutas_salida_benchmark(args_run, dataset_paths, cfg["protocol"])

                metricas = ejecutar_benchmark_tuned(
                    dataset_paths=dataset_paths,
                    funcion=normalizar_funcion(args_run.cec_funcid),
                    algoritmo=args_run.algorithm,
                    model_name=args_run.model,
                    param_grid=param_grid,
                    tuning_metric=args_run.tuning_metric,
                    inner_validation_ratio=args_run.inner_validation_ratio,
                    random_state=args_run.seed,
                    seed_selection_random_state=args_run.selection_seed,
                    truncar_convergencia=args_run.convergence_truncation,
                    store_tuning_results=args_run.store_tuning_results,
                    constructor_casos=cfg["constructor_casos"],
                    protocol=cfg["protocol"],
                    split_strategy=cfg["split_strategy"],
                )

                guardar_artefactos_modelo(
                    ruta_metricas=ruta_metricas,
                    ruta_runs_csv=ruta_runs_csv,
                    ruta_runs_json=ruta_runs_json,
                    metricas=metricas,
                    guardar_runs=True,
                )
                guardar_artefactos_bloques(
                    dir_subrogado=model_dir,
                    nombre_subrogado=args_run.model,
                    metricas=metricas,
                    guardar_runs=args_run.bloque_runs,
                )

                imprimir_resumen(metricas)


if __name__ == "__main__":
    main()
