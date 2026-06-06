from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocesado_de_datos.utils.path_utils import ALGORITMOS_MH, resolver_archivo_existente
from preprocesado_de_datos.utils.utils import cargar_dataset, inferir_seed
from surrogate_models.benchmark_utils.batches_eval_splitter import construir_casos_no_acumulativos
from surrogate_models.benchmark_utils.benchmark_paths import resolver_inputs_benchmark
from surrogate_models.benchmark_utils.evaluacion_offline import escalar_X
from surrogate_models.benchmark_utils.no_acumulativo.ajustar_hiperparametros import (
    dividir_train_validacion_interna,
    expand_param_grid,
)
from surrogate_models.feature_builders import construir_features
from surrogate_models.metrics import calcular_metricas
from surrogate_models.select_model import select_model


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Smoke test de grids RBF: prueba combinaciones de hiperparametros "
            "sobre una semilla y funcion concretas antes de lanzar el tuning completo."
        )
    )
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--algoritmo", required=True, choices=ALGORITMOS_MH)
    parser.add_argument("--funcion", required=True)
    parser.add_argument("--param-grid-json", required=True)
    parser.add_argument("--inputs", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-seeds", type=int, default=None)
    parser.add_argument("--seed-selection-random-state", type=int, default=42)
    parser.add_argument("--feature-mode", default="x", choices=["x", "x_div"])
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--future-validation", default="next", choices=["all", "next"])
    parser.add_argument("--inner-validation-ratio", type=float, default=0.20)
    parser.add_argument(
        "--batches",
        nargs="+",
        type=int,
        default=[1],
        help="Batches de entrenamiento a probar. Usa: --batches 1 2 3 4 para probar todos.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=2000,
        help="Maximo de muestras de entrenamiento interno usadas por configuracion.",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=500,
        help="Maximo de muestras de validacion interna usadas por configuracion.",
    )
    parser.add_argument(
        "--warnings-as-errors",
        action="store_true",
        help="Marcar como fallidas las configuraciones que emitan warnings.",
    )
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Ruta CSV de salida. Si no se indica, se crea junto al JSON de grid.",
    )
    return parser.parse_args()


def cargar_grid(path):
    path = resolver_archivo_existente(path, arg_name="param_grid_json")
    with path.open("r", encoding="utf-8") as fh:
        raw_grid = json.load(fh)
    return expand_param_grid(raw_grid), path


def elegir_dataset(args):
    dataset_paths = resolver_inputs_benchmark(args)
    if args.seed is None:
        return dataset_paths[0]
    for path in dataset_paths:
        if int(inferir_seed(path)) == int(args.seed):
            return path
    raise SystemExit(f"No se encontro seed={args.seed} para {args.algoritmo} {args.funcion}.")


def recortar_temporal(indices, max_samples, *, keep="start"):
    indices = np.asarray(indices, dtype=np.int64)
    if max_samples is None or max_samples < 1 or indices.size <= max_samples:
        return indices
    if keep == "end":
        return indices[-max_samples:]
    return indices[:max_samples]


def preparar_casos(dataset, args):
    casos = construir_casos_no_acumulativos(
        dataset,
        random_state=args.random_state,
        validation_scope=args.future_validation,
    )
    batches = set(int(batch) for batch in args.batches)
    casos = [caso for caso in casos if int(caso["batch_train"]) in batches]
    if not casos:
        raise SystemExit(f"No hay casos para batches={sorted(batches)}.")
    return casos


def probar_configuracion(params, x, y, caso, args):
    train_idx = np.asarray(caso["train_idx"], dtype=np.int64)
    inner_train_idx, inner_val_idx = dividir_train_validacion_interna(
        train_idx,
        args.inner_validation_ratio,
    )
    inner_train_idx = recortar_temporal(inner_train_idx, args.max_train_samples, keep="end")
    inner_val_idx = recortar_temporal(inner_val_idx, args.max_val_samples, keep="start")

    x_train = x[inner_train_idx]
    y_train = y[inner_train_idx]
    x_val = x[inner_val_idx]
    y_val = y[inner_val_idx]

    y_scaler = StandardScaler()
    y_train_fit = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

    captured_warnings = []
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("always")
        model = select_model("rbf", **params)
        model.fit(x_train, y_train_fit)
        y_pred = model.predict(x_val)
        captured_warnings = [
            f"{type(item.message).__name__}: {item.message}"
            for item in warning_records
        ]

    elapsed = time.perf_counter() - t0
    if args.warnings_as_errors and captured_warnings:
        raise RuntimeError(" | ".join(captured_warnings))

    y_pred = y_scaler.inverse_transform(np.asarray(y_pred).reshape(-1, 1)).ravel()
    metricas = calcular_metricas(y_val, y_pred)
    return {
        "ok": True,
        "error": "",
        "warnings": " | ".join(captured_warnings),
        "elapsed_s": elapsed,
        "spearman": metricas.get("spearman"),
        "nrmse": metricas.get("nrmse"),
        "nmae": metricas.get("nmae"),
        "n_train": int(x_train.shape[0]),
        "n_val": int(x_val.shape[0]),
    }


def main():
    args = parse_args()
    grid, grid_path = cargar_grid(args.param_grid_json)
    dataset_path = elegir_dataset(args)
    dataset = cargar_dataset(dataset_path)
    seed = int(inferir_seed(dataset_path))
    x = escalar_X(construir_features(dataset, args.feature_mode))
    y = np.asarray(dataset["fitness"], dtype=float).ravel()
    casos = preparar_casos(dataset, args)

    out_csv = (
        Path(args.out_csv)
        if args.out_csv
        else grid_path.with_name(f"{grid_path.stem}_smoke_{args.algoritmo}_{args.funcion}_s{seed}.csv")
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    total = len(grid) * len(casos)
    print(
        f"Smoke test RBF: {len(grid)} configuraciones x {len(casos)} batch(es) = {total} pruebas"
    )
    print(f"Dataset: {dataset_path}")
    print(f"Salida: {out_csv}")

    for i, params in enumerate(grid, 1):
        for caso in casos:
            row = {
                "config_id": i,
                "algoritmo": args.algoritmo,
                "funcion": args.funcion,
                "seed": seed,
                "batch_train": int(caso["batch_train"]),
                **params,
            }
            try:
                result = probar_configuracion(params, x, y, caso, args)
            except Exception as exc:  # noqa: BLE001 - el objetivo del smoke test es registrar fallos.
                result = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "warnings": "",
                    "elapsed_s": float("nan"),
                    "spearman": float("nan"),
                    "nrmse": float("nan"),
                    "nmae": float("nan"),
                    "n_train": args.max_train_samples,
                    "n_val": args.max_val_samples,
                }
            row.update(result)
            rows.append(row)

    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "ok",
        "config_id",
        "algoritmo",
        "funcion",
        "seed",
        "batch_train",
        "kernel",
        "epsilon",
        "smoothing",
        "neighbors",
        "degree",
        "spearman",
        "nrmse",
        "nmae",
        "elapsed_s",
        "warnings",
        "error",
    ]
    fieldnames = [key for key in preferred if key in fieldnames] + [
        key for key in fieldnames if key not in preferred
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_ok = sum(bool(row["ok"]) for row in rows)
    n_warn = sum(bool(row.get("warnings")) for row in rows)
    n_error = len(rows) - n_ok
    print(f"OK: {n_ok} | warnings: {n_warn} | errores: {n_error}")
    if n_error:
        errores = {}
        for row in rows:
            if not row["ok"]:
                errores[row["error"]] = errores.get(row["error"], 0) + 1
        print("Errores encontrados:")
        for error, count in sorted(errores.items(), key=lambda item: (-item[1], item[0])):
            print(f"  {count}x {error}")


if __name__ == "__main__":
    main()
