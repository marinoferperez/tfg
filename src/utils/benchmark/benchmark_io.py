"""Funciones de agregacion, guardado y mostrado de metricas del benchmark offline."""

import csv
from collections import defaultdict
from copy import deepcopy

import numpy as np

from src.utils.fs_utils import asegurar_directorio_padre
from src.utils.file_io import escribir_json
from src.utils.benchmark.surrogate_paths import rutas_bloque_modelo


def resumir_runs(metricas_runs):
    """
    Agrega las metricas individuales de cada run en un resumen estadistico.

    metricas_runs: lista de dicts con las metricas de cada run individual.
    """
    metricas_resumen = {}
    if not metricas_runs:
        for clave in ("mae", "nmae", "rmse", "nrmse", "spearman", "train_time_s", "predict_time_s"):
            metricas_resumen[clave] = float("nan")
            metricas_resumen[f"{clave}_std"] = float("nan")
        metricas_resumen["runs"] = []
        metricas_resumen["n_runs_evaluadas"] = 0
        metricas_resumen["n_seeds_evaluadas"] = 0
        metricas_resumen["n_train"] = 0
        metricas_resumen["n_test"] = 0
        return metricas_resumen

    for clave in ("mae", "nmae", "rmse", "nrmse", "train_time_s", "predict_time_s"):
        valores = np.asarray([run[clave] for run in metricas_runs], dtype=float)
        metricas_resumen[clave] = float(np.mean(valores))
        metricas_resumen[f"{clave}_std"] = float(np.std(valores))

    valores_spearman = np.asarray([run["spearman"] for run in metricas_runs], dtype=float)
    metricas_resumen["spearman"] = float(np.mean(valores_spearman))
    metricas_resumen["spearman_std"] = float(np.std(valores_spearman))

    metricas_resumen["runs"] = metricas_runs
    metricas_resumen["n_runs_evaluadas"] = int(len(metricas_runs))
    metricas_resumen["n_seeds_evaluadas"] = int(len({int(run["seed"]) for run in metricas_runs}))
    metricas_resumen["n_train"] = int(np.mean([run["n_train"] for run in metricas_runs]))
    metricas_resumen["n_test"] = int(np.mean([run["n_test"] for run in metricas_runs]))
    return metricas_resumen


def agrupar_runs_por_bloque(runs):
    """
    Agrupa los runs por bloque de entrenamiento.

    runs: lista de dicts de runs individuales con bloque_entrenamiento, train_pct_ini y train_pct_fin.
    """
    agrupados = defaultdict(list)
    for run in runs:
        clave = (int(run["bloque_entrenamiento"]), int(run["train_pct_ini"]), int(run["train_pct_fin"]))
        agrupados[clave].append(run)
    return dict(sorted(agrupados.items(), key=lambda item: item[0][0]))


def construir_metricas_bloque(metricas, runs_del_bloque):
    """
    Construye el dict de metricas para un bloque concreto a partir del resumen global.

    metricas: dict de metricas globales del experimento.
    runs_del_bloque: lista de runs pertenecientes al bloque.
    """
    primer_run = runs_del_bloque[0]
    metricas_bloque = resumir_runs(runs_del_bloque)
    metricas_bloque.update({
        "funcion": metricas["funcion"],
        "algoritmo": metricas["algoritmo"],
        "datasets": deepcopy(metricas["datasets"]),
        "n_datasets_entrada": metricas["n_datasets_entrada"],
        "model": metricas["model"],
        "feature_mode": metricas["feature_mode"],
        "model_params": deepcopy(metricas["model_params"]),
        "split_strategy": metricas["split_strategy"],
        "protocol": metricas["protocol"],
        "n_bloques": metricas["n_bloques"],
        "validation_ratio": metricas["validation_ratio"],
        "scale_features": metricas.get("scale_features"),
        "scale_target": metricas.get("scale_target"),
        "escalado": metricas["escalado"],
        "random_state": metricas["random_state"],
        "bloque_entrenamiento": int(primer_run["bloque_entrenamiento"]),
        "train_pct_ini": int(primer_run["train_pct_ini"]),
        "train_pct_fin": int(primer_run["train_pct_fin"]),
        "etiqueta_bloque": primer_run.get("etiqueta_bloque", f"{int(primer_run['train_pct_ini'])}-{int(primer_run['train_pct_fin'])}%"),
        "bloques_entrenamiento": primer_run.get("bloques_entrenamiento", str(primer_run["bloque_entrenamiento"])),
    })
    return metricas_bloque


def construir_payload_runs(metricas):
    """
    Construye el payload JSON con la lista detallada de runs.

    metricas: dict de metricas del experimento.
    """
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


def guardar_runs_csv(ruta_csv, runs):
    """
    Escribe los runs individuales en un CSV.

    ruta_csv: ruta del fichero de salida.
    runs: lista de dicts con las metricas por run.
    """
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


def guardar_artefactos_modelo(*, ruta_metricas, ruta_runs_csv, ruta_runs_json, metricas, guardar_runs=True):
    """
    Guarda el JSON de metricas y, opcionalmente, los CSVs y JSONs de runs.

    ruta_metricas: ruta del JSON de metricas del modelo.
    ruta_runs_csv: ruta del CSV de runs individuales.
    ruta_runs_json: ruta del JSON de runs; si es None no se genera.
    metricas: dict de metricas del experimento.
    guardar_runs: si True, escribe tambien los ficheros de runs.
    """
    escribir_json(ruta_metricas, metricas)
    if guardar_runs:
        guardar_runs_csv(ruta_runs_csv, metricas["runs"])
        if ruta_runs_json is not None:
            escribir_json(ruta_runs_json, construir_payload_runs(metricas))


def guardar_artefactos_bloques(*, dir_subrogado, nombre_subrogado, metricas, guardar_runs=False):
    """
    Guarda los artefactos de cada bloque en su propio subdirectorio.

    dir_subrogado: directorio raiz del modelo dentro del benchmark.
    nombre_subrogado: nombre del modelo subrogado.
    metricas: dict de metricas globales con la lista de runs.
    guardar_runs: si True, escribe tambien los CSVs de runs por bloque.
    """
    for (_bloque_entrenamiento, train_pct_ini, train_pct_fin), runs_bloque in agrupar_runs_por_bloque(
        metricas["runs"]
    ).items():
        _bloque_dir, ruta_metricas, ruta_runs_csv = rutas_bloque_modelo(dir_subrogado, nombre_subrogado, train_pct_ini, train_pct_fin)
        metricas_bloque = construir_metricas_bloque(metricas, runs_bloque)
        guardar_artefactos_modelo(
            ruta_metricas=ruta_metricas,
            ruta_runs_csv=ruta_runs_csv,
            ruta_runs_json=None,
            metricas=metricas_bloque,
            guardar_runs=guardar_runs,
        )


def imprimir_resumen(metricas):
    """
    Imprime por terminal un resumen de las metricas del benchmark.

    metricas: dict de metricas del experimento.
    """
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
    print(f"  train_time_s={metricas['train_time_s']:.6f} +- {metricas['train_time_s_std']:.6f}")
    print(f"  predict_time_s={metricas['predict_time_s']:.6f} +- {metricas['predict_time_s_std']:.6f}")
