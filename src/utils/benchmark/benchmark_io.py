"""Funciones de agregacion, guardado e impresion de metricas del benchmark offline."""

import csv
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np

from src.utils.fs_utils import asegurar_directorio_padre
from src.utils.file_io import escribir_json
from src.utils.benchmark.surrogate_paths import resolver_rutas_batch_modelo


def resumir_runs(metricas_runs):
    metricas_resumen = {}
    if not metricas_runs:
        for clave in (
            "mae",
            "nmae",
            "rmse",
            "nrmse",
            "spearman",
            "max_abs_error",
            "max_pct_error",
            "train_time_s",
            "predict_time_s",
        ):
            metricas_resumen[clave] = float("nan")
            metricas_resumen[f"{clave}_std"] = float("nan")
        metricas_resumen["spearman_n_validas"] = 0
        metricas_resumen["spearman_n_nan"] = 0
        metricas_resumen["runs"] = []
        metricas_resumen["n_runs_evaluadas"] = 0
        metricas_resumen["n_seeds_evaluadas"] = 0
        metricas_resumen["n_train"] = 0
        metricas_resumen["n_test"] = 0
        return metricas_resumen

    for clave in (
        "mae",
        "nmae",
        "rmse",
        "nrmse",
        "spearman",
        "max_abs_error",
        "max_pct_error",
        "train_time_s",
        "predict_time_s",
    ):
        valores = np.asarray([run[clave] for run in metricas_runs], dtype=float)
        metricas_resumen[clave] = float(np.mean(valores))
        metricas_resumen[f"{clave}_std"] = float(np.std(valores))

    # Spearman puede ser NaN en semillas concretas si la correlacion por rangos
    # no es definible. En el resumen agregamos solo sobre las semillas validas y
    # dejamos trazabilidad del numero de runs efectivamente usados.
    valores_spearman = np.asarray([run["spearman"] for run in metricas_runs], dtype=float)
    mascara_spearman_valida = np.isfinite(valores_spearman)
    metricas_resumen["spearman_n_validas"] = int(np.sum(mascara_spearman_valida))
    metricas_resumen["spearman_n_nan"] = int(np.sum(~mascara_spearman_valida))
    if metricas_resumen["spearman_n_validas"] == 0:
        metricas_resumen["spearman"] = float("nan")
        metricas_resumen["spearman_std"] = float("nan")
    else:
        valores_spearman_validos = valores_spearman[mascara_spearman_valida]
        metricas_resumen["spearman"] = float(np.mean(valores_spearman_validos))
        metricas_resumen["spearman_std"] = float(np.std(valores_spearman_validos))

    metricas_resumen["runs"] = metricas_runs
    metricas_resumen["n_runs_evaluadas"] = int(len(metricas_runs))
    metricas_resumen["n_seeds_evaluadas"] = int(len({int(run["seed"]) for run in metricas_runs}))
    metricas_resumen["n_train"] = int(np.mean([run["n_train"] for run in metricas_runs]))
    metricas_resumen["n_test"] = int(np.mean([run["n_test"] for run in metricas_runs]))
    return metricas_resumen


def agrupar_runs_por_batch(runs):
    agrupados = defaultdict(list)
    for run in runs:
        key = (
            int(run["batch_train"]),
            int(run["train_pct_ini"]),
            int(run["train_pct_fin"]),
        )
        agrupados[key].append(run)
    return dict(sorted(agrupados.items(), key=lambda item: item[0][0]))


def construir_metricas_batch(metricas, runs_batch):
    primer_run = runs_batch[0]
    metricas_batch = resumir_runs(runs_batch)
    metricas_batch.update(
        {
            "funcion": metricas["funcion"],
            "algoritmo": metricas["algoritmo"],
            "datasets": deepcopy(metricas["datasets"]),
            "n_datasets_entrada": metricas["n_datasets_entrada"],
            "model": metricas["model"],
            "feature_mode": metricas["feature_mode"],
            "model_params": deepcopy(metricas["model_params"]),
            "split_strategy": metricas["split_strategy"],
            "protocol": metricas["protocol"],
            "n_batches": metricas["n_batches"],
            "validation_ratio": metricas["validation_ratio"],
            "scale_features": metricas.get("scale_features"),
            "scale_target": metricas.get("scale_target"),
            "escalado": metricas["escalado"],
            "random_state": metricas["random_state"],
            "batch_train": int(primer_run["batch_train"]),
            "batch_train_last": int(primer_run.get("batch_train_last", primer_run["batch_train"])),
            "train_pct_ini": int(primer_run["train_pct_ini"]),
            "train_pct_fin": int(primer_run["train_pct_fin"]),
            "batch_label": primer_run.get(
                "batch_label",
                f"{int(primer_run['train_pct_ini'])}-{int(primer_run['train_pct_fin'])}%",
            ),
            "batches_train": primer_run.get("batches_train", str(primer_run["batch_train"])),
        }
    )
    if "sample_errors" in metricas:
        metricas_batch["sample_errors"] = [
            error
            for error in metricas["sample_errors"]
            if int(error["batch_train"]) == int(primer_run["batch_train"])
        ]
    return metricas_batch


def construir_payload_metricas(metricas):
    payload = dict(metricas)
    payload.pop("sample_errors", None)
    return payload


def construir_payload_runs(metricas):
    return {
        "funcion": metricas["funcion"],
        "algoritmo": metricas["algoritmo"],
        "model": metricas["model"],
        "feature_mode": metricas["feature_mode"],
        "split_strategy": metricas["split_strategy"],
        "protocol": metricas.get("protocol"),
        "validation_ratio": metricas["validation_ratio"],
        "scale_features": metricas.get("scale_features"),
        "scale_target": metricas.get("scale_target"),
        "n_runs_evaluadas": metricas["n_runs_evaluadas"],
        "runs": metricas["runs"],
    }


def construir_payload_errores(metricas):
    return {
        "funcion": metricas["funcion"],
        "algoritmo": metricas["algoritmo"],
        "model": metricas["model"],
        "feature_mode": metricas["feature_mode"],
        "split_strategy": metricas["split_strategy"],
        "protocol": metricas.get("protocol"),
        "validation_ratio": metricas["validation_ratio"],
        "scale_features": metricas.get("scale_features"),
        "scale_target": metricas.get("scale_target"),
        "sample_errors": metricas.get("sample_errors", []),
    }


def guardar_runs_csv(ruta_csv, runs):
    ruta_csv = asegurar_directorio_padre(ruta_csv)
    if not runs:
        ruta_csv.write_text("", encoding="utf-8")
        return

    runs_csv = []
    for run in runs:
        fila = dict(run)
        if "funcion" in fila:
            fila["cec_funcid"] = fila.pop("funcion")
        for clave in list(fila):
            if clave.startswith("max_"):
                fila.pop(clave, None)
        runs_csv.append(fila)

    fieldnames = []
    for fila in runs_csv:
        for clave in fila:
            if clave not in fieldnames:
                fieldnames.append(clave)
    with ruta_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(runs_csv)


def guardar_errores_csv(ruta_csv, errores):
    ruta_csv = asegurar_directorio_padre(ruta_csv)
    fieldnames = [
        "seed",
        "batch_train",
        "batch_train_last",
        "batch_label",
        "y_true",
        "y_pred",
    ]
    with ruta_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(errores)


def guardar_artefactos_modelo(
    *,
    ruta_metricas,
    ruta_runs_csv,
    ruta_runs_json,
    ruta_errores,
    metricas,
    guardar_runs=True,
):
    escribir_json(ruta_metricas, construir_payload_metricas(metricas))
    if guardar_runs:
        guardar_runs_csv(ruta_runs_csv, metricas["runs"])
        if ruta_runs_json is not None:
            escribir_json(ruta_runs_json, construir_payload_runs(metricas))

    if ruta_errores is not None:
        if Path(ruta_errores).suffix.lower() == ".csv":
            guardar_errores_csv(ruta_errores, metricas["sample_errors"])
        else:
            escribir_json(ruta_errores, construir_payload_errores(metricas))


def guardar_artefactos_batches(
    *,
    model_dir,
    model_name,
    ruta_errores_base,
    metricas,
    guardar_runs=False,
):
    errores_suffix = None
    if ruta_errores_base is not None:
        errores_suffix = Path(ruta_errores_base).suffix.lower() or ".json"

    for (_batch_train, train_pct_ini, train_pct_fin), runs_batch in agrupar_runs_por_batch(
        metricas["runs"]
    ).items():
        (
            _batch_dir,
            ruta_metricas,
            ruta_runs_csv,
            ruta_runs_json,
            ruta_errores,
        ) = resolver_rutas_batch_modelo(
            model_dir,
            model_name,
            train_pct_ini,
            train_pct_fin,
            errores_suffix=errores_suffix,
        )
        metricas_batch = construir_metricas_batch(metricas, runs_batch)
        guardar_artefactos_modelo(
            ruta_metricas=ruta_metricas,
            ruta_runs_csv=ruta_runs_csv,
            ruta_runs_json=None,
            ruta_errores=ruta_errores,
            metricas=metricas_batch,
            guardar_runs=guardar_runs,
        )


def imprimir_resumen(metricas):
    print("Resumen benchmark:")
    print(f"  funcion={metricas['funcion']}")
    print(f"  algoritmo={metricas['algoritmo']}")
    print(f"  model={metricas['model']}")
    print(f"  split_strategy={metricas['split_strategy']}")
    print(f"  n_datasets={metricas['n_datasets_entrada']}")
    print(f"  n_seeds_evaluadas={metricas['n_seeds_evaluadas']}")
    print(f"  n_runs_evaluadas={metricas['n_runs_evaluadas']}")
    print(f"  n_train={metricas['n_train']}")
    print(f"  n_test={metricas['n_test']}")
    print("Metricas:")
    print(f"  MAE={metricas['mae']:.6f} +- {metricas['mae_std']:.6f}")
    print(f"  nMAE={metricas['nmae']:.6f} +- {metricas['nmae_std']:.6f}")
    print(f"  RMSE={metricas['rmse']:.6f} +- {metricas['rmse_std']:.6f}")
    print(f"  nRMSE={metricas['nrmse']:.6f} +- {metricas['nrmse_std']:.6f}")
    print(f"  Spearman={metricas['spearman']:.6f} +- {metricas['spearman_std']:.6f}")
    if "spearman_n_validas" in metricas and "spearman_n_nan" in metricas:
        print(f"  spearman_n_validas={metricas['spearman_n_validas']} (NaN: {metricas['spearman_n_nan']})")
    print(f"  max_abs_error={metricas['max_abs_error']:.6f} +- {metricas['max_abs_error_std']:.6f}")
    print(f"  max_pct_error={metricas['max_pct_error']:.6f} +- {metricas['max_pct_error_std']:.6f}")
    print(f"  train_time_s={metricas['train_time_s']:.6f} +- {metricas['train_time_s_std']:.6f}")
    print(f"  predict_time_s={metricas['predict_time_s']:.6f} +- {metricas['predict_time_s_std']:.6f}")
