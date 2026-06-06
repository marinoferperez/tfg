from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocesado_de_datos.utils.path_utils import (
    ALGORITMOS_MH,
    normalizar_funcion,
    resolver_archivo_existente,
)
from preprocesado_de_datos.utils.utils import cargar_dataset, inferir_seed
from surrogate_models.benchmark_utils.batches_eval_splitter import (
    N_BATCHES,
    TOL_MEJORA_BATCH_ABS,
    TOL_MEJORA_BATCH_REL,
    VAL_RATIO_TRAIN,
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
from surrogate_models.feature_builders import construir_features
from surrogate_models.metrics import calcular_errores_por_muestra, calcular_metricas
from surrogate_models.select_model import select_model


# El escalado de X se aplica a todos los modelos porque es una normalizacion
# fija del dominio CEC2017, no una transformacion aprendida de los datos.
MODELOS_ARBOL = {"random_forest", "hgb", "xgboost"}
DOMAIN_BOUND = 100.0


def escalar_X(X):
    return X / DOMAIN_BOUND


def build_temporal_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--resultados-dir", dest="experiment_dir", required=True)
    parser.add_argument(
        "--benchmark-subdir",
        default="benchmarking",
        help=(
            "Subdirectorio relativo dentro del experimento donde guardar los resultados "
            "del benchmark. Por defecto: benchmarking"
        ),
    )
    parser.add_argument(
        "--algoritmo",
        default="all",
        choices=[*ALGORITMOS_MH, "all"],
        help="Metaheuristica evaluada. 'all' ejecuta AGE, DE y SHADE.",
    )
    parser.add_argument("--cec-funcid", dest="funcion", required=True)
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help="Lista opcional de datasets por seed. Si no se indica, se resuelven desde resultados-dir.",
    )
    parser.add_argument(
        "--modelo",
        dest="model",
        required=True,
        choices=["rbf", "svr", "mlp", "rsm", "random_forest", "hgb", "lasso", "xgboost"],
    )
    parser.add_argument("--seed", dest="random_state", type=int, default=42)
    parser.add_argument("--max-seeds", type=int, default=None)
    parser.add_argument(
        "--seed-selection-random-state",
        type=int,
        default=42,
        help=(
            "Semilla para seleccionar aleatoriamente el subconjunto de seeds cuando "
            "se usa --max-seeds. Default: 42"
        ),
    )
    parser.add_argument(
        "--convergence-truncation",
        action="store_true",
        default=False,
        help=(
            "Truncar el dataset en el último batch con mejora real del running best. "
            "Los batches posteriores a la última mejora se descartan antes de construir los casos."
        ),
    )
    parser.add_argument("--modelo-params-json", dest="model_params_json", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--runs-out", default=None)
    parser.add_argument("--runs-json-out", default=None)
    parser.add_argument(
        "--no-runs-json",
        action="store_true",
        help="No generar *_runs.json. El CSV de runs se mantiene como artefacto trazable principal.",
    )
    parser.add_argument(
        "--batch-runs",
        action="store_true",
        help=(
            "Generar tambien *_runs.csv por batch. Por defecto solo se guardan "
            "metricas por batch para poder construir resumenes."
        ),
    )
    parser.add_argument("--errors-out", default=None)
    parser.add_argument("--no-benchmark-summary", action="store_true")
    return parser


def cargar_model_kwargs(ruta_json):
    if ruta_json is None:
        return {}
    ruta = resolver_archivo_existente(ruta_json, arg_name="model_params_json")
    with ruta.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def ejecutar_benchmark_temporal(
    *,
    dataset_paths,
    funcion,
    algoritmo,
    model_name,
    feature_mode,
    model_kwargs,
    constructor_casos,
    protocol,
    split_strategy,
    random_state=42,
    seed_selection_random_state=None,
    collect_sample_errors=False,
    future_validation="all",
    truncar_convergencia=False,
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

        x = construir_features(dataset, feature_mode)
        y = np.asarray(dataset["fitness"], dtype=float).ravel()
        eval_id = np.asarray(dataset["eval_id"], dtype=np.int64)
        casos = constructor_casos(
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
            x_train = x[train_idx]
            x_val = x[val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]

            x_train = escalar_X(x_train)
            x_val = escalar_X(x_val)

            y_scaler = None
            y_train_fit = y_train
            if escalar_y:
                y_scaler = StandardScaler()
                y_train_fit = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

            model = select_model(model_name, **model_kwargs)

            t0 = time.perf_counter()
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
                }
            )
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
            "model_params": model_kwargs,
            "split_strategy": split_strategy,
            "future_validation": str(future_validation),
            "protocol": protocol,
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


def generar_resumen_global(benchmark_dir, resumen_script_path):
    subprocess.run(
        [
            sys.executable,
            str(resumen_script_path),
            "--benchmark-dir",
            str(benchmark_dir),
        ],
        check=True,
    )


def main_temporal(
    *,
    protocol,
    split_strategy,
    constructor_casos,
    resumen_script_name,
    description,
):
    parser = build_temporal_parser(description)
    args = parser.parse_args()
    algoritmos = ALGORITMOS_MH if args.algoritmo == "all" else (args.algoritmo,)
    for algoritmo in algoritmos:
        args_algoritmo = argparse.Namespace(**vars(args))
        args_algoritmo.algoritmo = algoritmo
        dataset_paths = resolver_inputs_benchmark(args_algoritmo)
        (
            benchmark_dir,
            model_dir,
            ruta_metricas,
            ruta_runs_csv,
            ruta_runs_json,
            ruta_errores,
        ) = resolver_rutas_salida_benchmark(args_algoritmo, dataset_paths, protocol)

        metricas = ejecutar_benchmark_temporal(
            dataset_paths=dataset_paths,
            funcion=normalizar_funcion(args_algoritmo.funcion),
            algoritmo=args_algoritmo.algoritmo,
            model_name=args_algoritmo.model,
            feature_mode="x",
            model_kwargs=cargar_model_kwargs(args_algoritmo.model_params_json),
            constructor_casos=constructor_casos,
            protocol=protocol,
            split_strategy=split_strategy,
            random_state=args_algoritmo.random_state,
            seed_selection_random_state=args_algoritmo.seed_selection_random_state,
            collect_sample_errors=(ruta_errores is not None),
            future_validation="next",
            truncar_convergencia=args_algoritmo.convergence_truncation,
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
            model_name=args_algoritmo.model,
            ruta_errores_base=ruta_errores,
            metricas=metricas,
            guardar_runs=args_algoritmo.batch_runs,
        )

        if not args_algoritmo.no_benchmark_summary:
            resumen_script_path = Path(__file__).resolve().parent / protocol / resumen_script_name
            generar_resumen_global(benchmark_dir, resumen_script_path)

        imprimir_resumen(metricas)
