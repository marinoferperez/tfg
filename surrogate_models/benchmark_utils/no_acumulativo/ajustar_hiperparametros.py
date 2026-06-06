from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocesado_de_datos.utils.path_utils import ALGORITMOS_MH, normalizar_funcion, resolver_archivo_existente
from preprocesado_de_datos.utils.utils import cargar_dataset, inferir_seed
from surrogate_models.benchmark_utils.batches_eval_splitter import (
    N_BATCHES,
    TOL_MEJORA_BATCH_ABS,
    TOL_MEJORA_BATCH_REL,
    VAL_RATIO_TRAIN,
    construir_casos_no_acumulativos,
    truncar_por_convergencia,
)
from surrogate_models.benchmark_utils.benchmark_io import (
    guardar_artefactos_batches,
    guardar_artefactos_modelo,
    imprimir_resumen,
    resumir_runs,
)
from surrogate_models.benchmark_utils.benchmark_paths import (
    resolver_inputs_benchmark,
    resolver_rutas_salida_benchmark,
)
from surrogate_models.benchmark_utils.evaluacion_offline import MODELOS_ARBOL, escalar_X
from surrogate_models.feature_builders import construir_features
from surrogate_models.metrics import calcular_errores_por_muestra, calcular_metricas
from surrogate_models.select_model import select_model


METRICAS_MAXIMIZAR = {"spearman"}
METRICAS_MINIMIZAR = {"mae", "nmae", "rmse", "nrmse"}
CEC2017_FUNCIONES = tuple(f"f{i}" for i in range(1, 31))
MODELOS_AJUSTE = ("rbf", "svr", "mlp", "rsm", "random_forest", "hgb", "lasso", "xgboost")


def grid_rsm_default():
    return [
        {"degree": 1},
        {"degree": 2},
        {"degree": 3},
    ]


def grid_rbf_default():
    grid = []
    for neighbors, smoothing in itertools.product((25, 50, 100), (1e-3, 1e-2)):
        grid.append(
            {
                "neighbors": neighbors,
                "smoothing": smoothing,
                "kernel": "linear",
                "degree": -1,
            }
        )
    for neighbors, smoothing in itertools.product((50, 100), (1e-3, 1e-2)):
        grid.append(
            {
                "neighbors": neighbors,
                "smoothing": smoothing,
                "kernel": "gaussian",
                "degree": -1,
                "epsilon": 1.0,
            }
        )
    return grid


def default_param_grid(model_name):
    if model_name == "rsm":
        return grid_rsm_default()
    if model_name == "rbf":
        return grid_rbf_default()
    if model_name in MODELOS_AJUSTE:
        return [{}]
    raise ValueError(f"Modelo no soportado para ajuste: {model_name!r}.")


def expand_param_grid(raw_grid):
    if isinstance(raw_grid, list):
        return [dict(item) for item in raw_grid]
    if isinstance(raw_grid, dict):
        keys = list(raw_grid.keys())
        values = [raw_grid[key] if isinstance(raw_grid[key], list) else [raw_grid[key]] for key in keys]
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    raise ValueError("El grid de parametros debe ser una lista de diccionarios o un diccionario de listas.")


def expandir_funciones(funcion_arg):
    txt = str(funcion_arg).strip().lower()
    if txt == "all":
        return CEC2017_FUNCIONES
    partes = [parte.strip() for parte in txt.split(",") if parte.strip()]
    if len(partes) > 1:
        return tuple(normalizar_funcion(parte) for parte in partes)
    return (normalizar_funcion(txt),)


def expandir_modelos(model_arg):
    txt = str(model_arg).strip().lower()
    if txt == "all":
        return MODELOS_AJUSTE
    partes = [parte.strip() for parte in txt.split(",") if parte.strip()]
    modelos = partes if len(partes) > 1 else [txt]
    invalidos = [modelo for modelo in modelos if modelo not in MODELOS_AJUSTE]
    if invalidos:
        raise ValueError(f"Modelos no soportados para ajuste: {', '.join(invalidos)}")
    return tuple(modelos)


def cargar_param_grid(model_name, ruta_json):
    if ruta_json is None:
        return default_param_grid(model_name)
    ruta = resolver_archivo_existente(ruta_json, arg_name="param_grid_json")
    with ruta.open("r", encoding="utf-8") as fh:
        return expand_param_grid(json.load(fh))


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark temporal no acumulativo con ajuste interno de hiperparametros "
            "dentro de cada bloque de entrenamiento."
        )
    )
    parser.add_argument("--trayectoria-dir", dest="experiment_dir", required=True)
    parser.add_argument(
        "--ajuste-resultados-dir",
        dest="benchmark_subdir",
        default="resultados_benchmark_surrogates_offline_ajuste",
    )
    parser.add_argument("--algoritmo", default="all", choices=[*ALGORITMOS_MH, "all"])
    parser.add_argument("--cec-funcid", dest="funcion", required=True)
    parser.add_argument("--inputs", nargs="+", default=None)
    parser.add_argument("--modelo", dest="model", default="all")
    parser.add_argument("--seed", dest="random_state", type=int, default=42)
    parser.add_argument("--max-seeds", type=int, default=None)
    parser.add_argument("--seed-selection-random-state", type=int, default=42)
    parser.add_argument("--convergence-truncation", action="store_true", default=False)
    parser.add_argument("--param-grid-json", default=None)
    parser.add_argument("--metrica-ajuste", dest="tuning_metric", default="spearman", choices=sorted(METRICAS_MAXIMIZAR | METRICAS_MINIMIZAR))
    parser.add_argument(
        "--validacion-ratio",
        dest="inner_validation_ratio",
        type=float,
        default=0.20,
        help="Fraccion final del bloque de entrenamiento usada como validacion interna.",
    )
    parser.add_argument(
        "--store-tuning-results",
        action="store_true",
        help="Guardar el detalle completo de todos los candidatos del grid en cada run.",
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--runs-out", default=None)
    parser.add_argument("--runs-json-out", default=None)
    parser.add_argument("--no-runs-json", action="store_true")
    parser.add_argument("--batch-runs", action="store_true")
    parser.add_argument("--errors-out", default=None)
    parser.add_argument("--no-benchmark-summary", action="store_true")
    return parser


def dividir_train_validacion_interna(train_idx, ratio):
    train_idx = np.asarray(train_idx, dtype=np.int64)
    if not 0.0 < float(ratio) < 0.5:
        raise ValueError("inner-validation-ratio debe estar entre 0 y 0.5.")
    n_val = int(np.floor(train_idx.size * float(ratio)))
    if n_val < 1 or train_idx.size - n_val < 1:
        raise ValueError("No hay suficientes muestras para validacion interna.")
    return train_idx[:-n_val], train_idx[-n_val:]


def ajustar_y_predecir(model_name, params, x_train, y_train, x_val, *, escalar_y):
    t0 = time.perf_counter()
    y_scaler = None
    y_train_fit = y_train
    if escalar_y:
        y_scaler = StandardScaler()
        y_train_fit = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

    model = select_model(model_name, **params)
    model.fit(x_train, y_train_fit)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred = model.predict(x_val)
    predict_time = time.perf_counter() - t1

    if y_scaler is not None:
        y_pred = y_scaler.inverse_transform(np.asarray(y_pred).reshape(-1, 1)).ravel()
    return np.asarray(y_pred, dtype=float).ravel(), float(train_time), float(predict_time)


def es_mejor(score, best_score, metric):
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
    escalar_y,
):
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
            y_pred, train_time, predict_time = ajustar_y_predecir(
                model_name,
                params,
                x_inner_train,
                y_inner_train,
                x_inner_val,
                escalar_y=escalar_y,
            )
            metricas = calcular_metricas(y_inner_val, y_pred)
            score = float(metricas[metric])
            error = None
        except Exception as exc:  # noqa: BLE001 - se registra el fallo y se prueba el siguiente candidato.
            score = float("nan")
            metricas = {}
            train_time = float("nan")
            predict_time = float("nan")
            error = f"{type(exc).__name__}: {exc}"

        resultados.append(
            {
                "params": params,
                "score": score,
                "metric": metric,
                "error": error,
                "metricas": metricas,
                "train_time_s": train_time,
                "predict_time_s": predict_time,
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
    feature_mode,
    param_grid,
    tuning_metric,
    inner_validation_ratio,
    random_state,
    seed_selection_random_state,
    collect_sample_errors,
    future_validation,
    truncar_convergencia,
    store_tuning_results,
):
    escalar_y = model_name not in MODELOS_ARBOL
    metricas_runs = []
    sample_errors = []
    seeds_sin_casos = []
    convergencia_por_seed = {}

    for dataset_path in dataset_paths:
        dataset = cargar_dataset(dataset_path)
        seed_dataset = int(inferir_seed(dataset_path))

        convergencia_ultimo_batch = N_BATCHES
        convergencia_fraccion = 1.0
        if truncar_convergencia:
            dataset, convergencia_ultimo_batch, convergencia_fraccion = truncar_por_convergencia(dataset)
        convergencia_por_seed[seed_dataset] = {
            "ultimo_batch_informativo": int(convergencia_ultimo_batch),
            "fraccion_retenida": float(convergencia_fraccion),
        }

        x = escalar_X(construir_features(dataset, feature_mode))
        y = np.asarray(dataset["fitness"], dtype=float).ravel()
        eval_id = np.asarray(dataset["eval_id"], dtype=np.int64)
        casos = construir_casos_no_acumulativos(
            dataset,
            random_state=random_state,
            validation_scope=future_validation,
        )
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
                escalar_y=escalar_y,
            )

            x_train = x[train_idx]
            x_val = x[val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]

            t0 = time.perf_counter()
            y_scaler = None
            y_train_fit = y_train
            if escalar_y:
                y_scaler = StandardScaler()
                y_train_fit = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

            model = select_model(model_name, **best_params)
            model.fit(x_train, y_train_fit)
            train_time = time.perf_counter() - t0

            t1 = time.perf_counter()
            y_pred = model.predict(x_val)
            pred_time = time.perf_counter() - t1
            if y_scaler is not None:
                y_pred = y_scaler.inverse_transform(np.asarray(y_pred).reshape(-1, 1)).ravel()

            metricas_run = calcular_metricas(y_val, y_pred)
            metricas_run.update(
                {
                    "funcion": str(funcion),
                    "algoritmo": str(algoritmo),
                    "seed": int(caso["seed"]),
                    "batch_train": int(caso["batch_train"]),
                    "batch_train_last": int(caso.get("batch_train_last", caso["batch_train"])),
                    "train_pct_ini": int(caso["train_pct_ini"]),
                    "train_pct_fin": int(caso["train_pct_fin"]),
                    "batch_label": str(caso["batch_label"]),
                    "batches_train": ",".join(str(v) for v in caso.get("batches_train", [])),
                    "batches_futuros": ",".join(str(v) for v in caso["batches_futuros"]),
                    "n_train": int(caso["n_train"]),
                    "n_test": int(caso["n_val"]),
                    "train_idx_hash": str(caso["train_idx_hash"]),
                    "val_idx_hash": str(caso["val_idx_hash"]),
                    "eval_id_train_min": int(np.min(eval_id[train_idx])),
                    "eval_id_train_max": int(np.max(eval_id[train_idx])),
                    "eval_id_val_min": int(np.min(eval_id[val_idx])),
                    "eval_id_val_max": int(np.max(eval_id[val_idx])),
                    "train_time_s": float(train_time),
                    "predict_time_s": float(pred_time),
                    "convergencia_ultimo_batch": int(convergencia_ultimo_batch),
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

            if collect_sample_errors:
                error_abs, error_pct = calcular_errores_por_muestra(y_val, y_pred)
                for i in range(len(y_val)):
                    sample_errors.append(
                        {
                            "seed": int(caso["seed"]),
                            "batch_train": int(caso["batch_train"]),
                            "batch_train_last": int(caso.get("batch_train_last", caso["batch_train"])),
                            "batch_label": str(caso["batch_label"]),
                            "y_true": float(y_val[i]),
                            "y_pred": float(y_pred[i]),
                            "error_abs": float(error_abs[i]),
                            "error_pct": None if np.isnan(error_pct[i]) else float(error_pct[i]),
                        }
                    )

    metricas = resumir_runs(metricas_runs)
    metricas.update(
        {
            "funcion": str(funcion),
            "algoritmo": str(algoritmo),
            "datasets": [str(Path(path).resolve()) for path in dataset_paths],
            "selected_seeds": [int(inferir_seed(path)) for path in dataset_paths],
            "n_datasets_entrada": int(len(dataset_paths)),
            "model": model_name,
            "feature_mode": feature_mode,
            "model_params": {
                "tuning": True,
                "tuning_metric": tuning_metric,
                "inner_validation_ratio": float(inner_validation_ratio),
                "param_grid": param_grid,
            },
            "split_strategy": "temporal_no_acumulativo_futuro_tuned",
            "future_validation": str(future_validation),
            "protocol": "no_acumulativo",
            "n_batches": int(N_BATCHES),
            "validation_ratio": float(VAL_RATIO_TRAIN),
            "scale_features": True,
            "scale_target": bool(escalar_y),
            "escalado": True,
            "random_state": int(random_state),
            "seed_selection_random_state": (
                None if seed_selection_random_state is None else int(seed_selection_random_state)
            ),
            "convergencia_criterio": (
                "mejora_practica_por_batch"
                if truncar_convergencia
                else ""
            ),
            "convergencia_tol_abs": (
                float(TOL_MEJORA_BATCH_ABS)
                if truncar_convergencia
                else None
            ),
            "convergencia_tol_rel": (
                float(TOL_MEJORA_BATCH_REL)
                if truncar_convergencia
                else None
            ),
            "seeds_sin_casos_validacion": [int(seed) for seed in seeds_sin_casos],
            "n_seeds_sin_casos_validacion": int(len(seeds_sin_casos)),
            "convergencia_por_seed": convergencia_por_seed,
        }
    )
    if collect_sample_errors:
        metricas["sample_errors"] = sample_errors
    return metricas


def generar_resumen_global(benchmark_dir):
    resumen_script_path = Path(__file__).resolve().parent / "resumir_no_acumulativo.py"
    subprocess.run(
        [
            sys.executable,
            str(resumen_script_path),
            "--benchmark-dir",
            str(benchmark_dir),
        ],
        check=True,
    )


def main():
    parser = build_parser()
    args = parser.parse_args()
    funciones = expandir_funciones(args.funcion)
    algoritmos = ALGORITMOS_MH if args.algoritmo == "all" else (args.algoritmo,)
    modelos = expandir_modelos(args.model)
    ejecucion_multiple = len(funciones) > 1 or len(algoritmos) > 1 or len(modelos) > 1
    salidas_explicitas = any(
        (args.out, args.runs_out, args.runs_json_out, args.errors_out)
    )
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
                args_run.funcion = funcion
                args_run.algoritmo = algoritmo
                args_run.model = modelo
                param_grid = cargar_param_grid(args_run.model, args_run.param_grid_json)
                if not param_grid:
                    raise SystemExit("El grid de hiperparametros esta vacio.")
                dataset_paths = resolver_inputs_benchmark(args_run)
                (
                    benchmark_dir,
                    model_dir,
                    ruta_metricas,
                    ruta_runs_csv,
                    ruta_runs_json,
                    ruta_errores,
                ) = resolver_rutas_salida_benchmark(args_run, dataset_paths, "no_acumulativo")

                metricas = ejecutar_benchmark_tuned(
                    dataset_paths=dataset_paths,
                    funcion=normalizar_funcion(args_run.funcion),
                    algoritmo=args_run.algoritmo,
                    model_name=args_run.model,
                    feature_mode="x",
                    param_grid=param_grid,
                    tuning_metric=args_run.tuning_metric,
                    inner_validation_ratio=args_run.inner_validation_ratio,
                    random_state=args_run.random_state,
                    seed_selection_random_state=args_run.seed_selection_random_state,
                    collect_sample_errors=(ruta_errores is not None),
                    future_validation="next",
                    truncar_convergencia=args_run.convergence_truncation,
                    store_tuning_results=args_run.store_tuning_results,
                )

                guardar_artefactos_modelo(
                    ruta_metricas=ruta_metricas,
                    ruta_runs_csv=ruta_runs_csv,
                    ruta_runs_json=ruta_runs_json,
                    ruta_errores=ruta_errores,
                    metricas=metricas,
                    guardar_runs=True,
                )
                guardar_artefactos_batches(
                    model_dir=model_dir,
                    model_name=args_run.model,
                    ruta_errores_base=ruta_errores,
                    metricas=metricas,
                    guardar_runs=args_run.batch_runs,
                )

                if not args_run.no_benchmark_summary:
                    generar_resumen_global(benchmark_dir)

                imprimir_resumen(metricas)


if __name__ == "__main__":
    main()
