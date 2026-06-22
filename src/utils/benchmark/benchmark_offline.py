"""Logica central del benchmark offline: carga de datasets, entrenamiento y evaluacion por bloques."""

import json
import time
from pathlib import Path

import numpy as np

from src.utils.fs_utils import resolver_archivo_existente
from src.utils.dataset_utils import cargar_dataset, inferir_seed
from src.surrogates.preprocessing.scaling import (
    MODELOS_ARBOL,
    ajustar_y,
    construir_escalador_y,
    escalar_X,
    invertir_y,
)
from src.utils.benchmark.blocks_eval_splitter import (
    N_BLOQUES,
    TOL_MEJORA_BLOQUE_ABS,
    TOL_MEJORA_BLOQUE_REL,
    VAL_RATIO_TRAIN,
    truncar_por_convergencia,
)
from src.utils.benchmark.benchmark_io import resumir_runs
from src.surrogates.evaluation.metrics import calcular_metricas
from src.surrogates.select_model import select_model


def cargar_hiper_subrogado(ruta_json):
    """
    Carga los hiperparametros del modelo desde un JSON, o devuelve {} si no se indica.

    ruta_json: ruta al fichero JSON con los hiperparametros; None devuelve un dict vacio.
    """
    if ruta_json is None:
        return {}
    ruta = resolver_archivo_existente(ruta_json, arg_name="model_params_json")
    with ruta.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def ejecutar_benchmark_temporal(*, dataset_paths, funcion, algoritmo, nombre_subrogado, hiper_subrogado, constructor_casos, protocolo, estrategia_split, random_state=42, seed_selection_random_state=None, truncar_convergencia=False):
    """
    Ejecuta el benchmark temporal sobre los datasets indicados.

    dataset_paths: lista de rutas a los ficheros HDF5 del dataset.
    funcion: nombre o identificador de la funcion CEC evaluada.
    algoritmo: nombre del algoritmo que genero los datasets.
    nombre_subrogado: nombre del modelo subrogado a evaluar.
    hiper_subrogado: hiperparametros del modelo.
    constructor_casos: callable que genera los casos de entrenamiento/validacion.
    protocolo: protocoloo de evaluacion (p.ej. "no_acumulativo").
    estrategia_split: estrategia de particion usada.
    random_state: semilla base para el muestreo de validacion.
    seed_selection_random_state: semilla para la seleccion aleatoria de seeds; None usa todas.
    truncar_convergencia: si True, recorta cada dataset al ultimo bloque informativo.
    """
    # los modelos de arbol son invariantes a escala, no necesitan escalar Y
    escalar_y = nombre_subrogado not in MODELOS_ARBOL
    metricas_runs = []
    seeds_sin_casos = []
    convergencia_por_seed = {}

    for dataset_path in dataset_paths:
        dataset = cargar_dataset(dataset_path)
        seed_dataset = int(inferir_seed(dataset_path))

        # si no se trunca, se considera que todos los bloques son informativos
        convergencia_ultimo_bloque = N_BLOQUES
        convergencia_fraccion = 1.0
        if truncar_convergencia:
            dataset, convergencia_ultimo_bloque, convergencia_fraccion = truncar_por_convergencia(dataset)
        convergencia_por_seed[seed_dataset] = {
            "ultimo_bloque_informativo": int(convergencia_ultimo_bloque),
            "fraccion_retenida": float(convergencia_fraccion),
        }

        x = np.asarray(dataset["x"], dtype=float)
        y = np.asarray(dataset["fitness"], dtype=float).ravel()
        # eval_id se necesita para anotar el rango temporal de cada split en las metricas
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

            # el scaler se ajusta solo sobre train para evitar data leakage
            y_scaler = construir_escalador_y(nombre_subrogado)
            y_train_fit = ajustar_y(y_scaler, y_train)

            model = select_model(nombre_subrogado, **hiper_subrogado)

            t0 = time.perf_counter()
            model.fit(x_train, y_train_fit)
            tiempo_entrenamiento = time.perf_counter() - t0

            t1 = time.perf_counter()
            y_pred = model.predict(x_val)
            tiempo_prediccion = time.perf_counter() - t1

            # se invierten los escalados para valores reales
            y_pred = invertir_y(y_scaler, y_pred)

            metricas_run = calcular_metricas(y_val, y_pred)
            metricas_run.update({
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
            })
            metricas_runs.append(metricas_run)

    metricas = resumir_runs(metricas_runs)
    metricas.update({
        "funcion": str(funcion),
        "algoritmo": str(algoritmo),
        "datasets": [str(Path(path).resolve()) for path in dataset_paths],
        "selected_seeds": [int(inferir_seed(path)) for path in dataset_paths],
        "n_datasets_entrada": int(len(dataset_paths)),
        "model": nombre_subrogado,
        "feature_mode": "x",
        "model_params": hiper_subrogado,
        "split_strategy": estrategia_split,
        "protocol": protocolo,
        "n_bloques": int(N_BLOQUES),
        "validation_ratio": float(VAL_RATIO_TRAIN),
        "scale_features": True,
        "scale_target": bool(escalar_y),
        "escalado": True,
        "random_state": int(random_state),
        "seed_selection_random_state": None if seed_selection_random_state is None else int(seed_selection_random_state),
        "convergencia_criterio": "mejora_practica_por_bloque" if truncar_convergencia else "",
        "convergencia_tol_abs": float(TOL_MEJORA_BLOQUE_ABS) if truncar_convergencia else None,
        "convergencia_tol_rel": float(TOL_MEJORA_BLOQUE_REL) if truncar_convergencia else None,
        "seeds_sin_casos_validacion": [int(seed) for seed in seeds_sin_casos],
        "n_seeds_sin_casos_validacion": int(len(seeds_sin_casos)),
        "convergencia_por_seed": convergencia_por_seed,
    })
    return metricas
