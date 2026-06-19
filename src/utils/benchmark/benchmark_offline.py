"""Logica central del benchmark offline: carga de datasets, entrenamiento y evaluacion por batches."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.fs_utils import resolver_archivo_existente
from src.utils.dataset_utils import cargar_dataset, inferir_seed
from src.surrogates.preprocessing.scaling import (
    MODELOS_ARBOL,
    ajustar_y,
    construir_escalador_y,
    escalar_X,
    invertir_y,
)
from src.utils.benchmark.batches_eval_splitter import (
    N_BATCHES,
    TOL_MEJORA_BATCH_ABS,
    TOL_MEJORA_BATCH_REL,
    VAL_RATIO_TRAIN,
    truncar_por_convergencia,
)
from src.utils.benchmark.benchmark_io import resumir_runs
from src.surrogates.evaluation.metrics import calcular_errores_por_muestra, calcular_metricas
from src.surrogates.select_model import select_model


def cargar_model_kwargs(ruta_json):
    """Carga hiperparametros del modelo desde un JSON, o devuelve {} si no se indica."""
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
    model_kwargs,
    constructor_casos,
    protocol,
    split_strategy,
    random_state=42,
    seed_selection_random_state=None,
    collect_sample_errors=False,
    truncar_convergencia=False,
):
    """
    Ejecuta el benchmark temporal sobre los datasets indicados.

    Entrena y valida el modelo en cada caso generado por constructor_casos,
    agrega metricas por seed y devuelve el resumen completo.
    """
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

        x = np.asarray(dataset["x"], dtype=float)
        y = np.asarray(dataset["fitness"], dtype=float).ravel()
        eval_id = np.asarray(dataset["eval_id"], dtype=np.int64)
        casos = constructor_casos(dataset, random_state=random_state)
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

            y_scaler = construir_escalador_y(model_name)
            y_train_fit = ajustar_y(y_scaler, y_train)

            model = select_model(model_name, **model_kwargs)

            t0 = time.perf_counter()
            model.fit(x_train, y_train_fit)
            train_time = time.perf_counter() - t0

            t1 = time.perf_counter()
            y_pred = model.predict(x_val)
            pred_time = time.perf_counter() - t1

            y_pred = invertir_y(y_scaler, y_pred)

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
                    "batch_validacion": int(caso["batch_validacion"]),
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
            "feature_mode": "x",
            "model_params": model_kwargs,
            "split_strategy": split_strategy,
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
                "mejora_practica_por_batch" if truncar_convergencia else ""
            ),
            "convergencia_tol_abs": (
                float(TOL_MEJORA_BATCH_ABS) if truncar_convergencia else None
            ),
            "convergencia_tol_rel": (
                float(TOL_MEJORA_BATCH_REL) if truncar_convergencia else None
            ),
            "seeds_sin_casos_validacion": [int(seed) for seed in seeds_sin_casos],
            "n_seeds_sin_casos_validacion": int(len(seeds_sin_casos)),
            "convergencia_por_seed": convergencia_por_seed,
        }
    )
    if collect_sample_errors:
        metricas["sample_errors"] = sample_errors
    return metricas
