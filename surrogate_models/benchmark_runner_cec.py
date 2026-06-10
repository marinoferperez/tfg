import argparse
import csv
import json
import subprocess
import sys
import time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.preprocessing import StandardScaler
from surrogate_models.select_model import select_model
from preprocesado_de_datos.utils.path_utils import (
    ALGORITMOS_MH,
    escribir_json,
    inferir_benchmark_dir_desde_candidatos,
    inferir_directorio_modelo,
    resolver_archivo_existente,
    resolver_inputs_experimento,
)
from preprocesado_de_datos.utils.utils import cargar_dataset, inferir_seed
from surrogate_models.metrics import calcular_errores_por_muestra, calcular_metricas
from surrogate_models.splitters import split_por_run_aleatorio

# El escalado de X se aplica a todos los modelos porque es una normalizacion
# fija del dominio CEC2017, no una transformacion aprendida de los datos.
MODELOS_ARBOL = {"random_forest", "hgb", "xgboost"}

# Dominio CEC2017: [-100, 100] → normalizar a [-1, 1] dividiendo por 100.
# Esta transformación es FIJA (no se fittea sobre muestras).
DOMAIN_BOUND = 100.0

def escalar_X(X):
    """Escalado por dominio: [-100, 100] → [-1, 1]. Invariante a las muestras."""
    return X / DOMAIN_BOUND

def resumir_runs(metricas_runs):
    if not metricas_runs:
        raise ValueError("No hay metricas de runs para resumir.")
    metricas_resumen = {}
    for clave in (
        "mae",
        "rmse",
        "spearman",
        "max_abs_error",
        "max_pct_error",
        "train_time_s",
        "predict_time_s",
    ):
        valores = np.asarray([run[clave] for run in metricas_runs], dtype=float)
        metricas_resumen[clave] = float(np.mean(valores))
        metricas_resumen[f"{clave}_std"] = float(np.std(valores))
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
    metricas_resumen["n_train"] = int(np.mean([run["n_train"] for run in metricas_runs]))
    metricas_resumen["n_test"] = int(np.mean([run["n_test"] for run in metricas_runs]))
    return metricas_resumen


def ordenar_paths_por_seed(dataset_paths):
    def clave(path):
        try:
            return (0, inferir_seed(path))
        except Exception:
            return (1, str(path))

    return sorted((Path(path) for path in dataset_paths), key=clave)


def ejecutar_benchmark(
    dataset_paths,
    model_name,
    feature_mode,
    model_kwargs,
    random_state=42,
    train_ratio=0.7,
    max_seeds=None,
    collect_sample_errors=False,
):
    escalar_y = model_name not in MODELOS_ARBOL
    metricas_runs = []
    sample_errors = []

    dataset_paths = [Path(path) for path in dataset_paths]
    max_seeds_split = max_seeds if len(dataset_paths) == 1 else None

    for dataset_path in dataset_paths:
        dataset = cargar_dataset(dataset_path)
        x = np.asarray(dataset["x"], dtype=float)
        y = np.asarray(dataset["fitness"], dtype=float).ravel()

        splits = split_por_run_aleatorio(
            dataset,
            random_state=random_state,
            train_ratio=train_ratio,
            max_seeds=max_seeds_split,
        )

        for split in splits:
            train_idx = split["train_idx"]
            test_idx = split["test_idx"]

            x_train = x[train_idx]
            x_test = x[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            x_train = escalar_X(x_train)
            x_test = escalar_X(x_test)

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
            y_pred = model.predict(x_test)
            pred_time = time.perf_counter() - t1

            if y_scaler is not None:
                y_pred = y_scaler.inverse_transform(np.asarray(y_pred).reshape(-1, 1)).ravel()

            metricas_run = calcular_metricas(y_test, y_pred)
            metricas_run.update({
                "seed": int(split["seed"]),
                "n_train": int(x_train.shape[0]),
                "n_test": int(x_test.shape[0]),
                "train_time_s": float(train_time),
                "predict_time_s": float(pred_time),
            })
            metricas_runs.append(metricas_run)

            if collect_sample_errors:
                error_abs, error_pct = calcular_errores_por_muestra(y_test, y_pred)
                for i in range(len(y_test)):
                    sample_errors.append({
                        "seed": int(split["seed"]),
                        "y_true": float(y_test[i]),
                        "y_pred": float(y_pred[i]),
                        "error_abs": float(error_abs[i]),
                        "error_pct": None if np.isnan(error_pct[i]) else float(error_pct[i]),
                    })

    metricas = resumir_runs(metricas_runs)
    metricas.update({
        "dataset": str(dataset_paths[0]) if len(dataset_paths) == 1 else None,
        "datasets": [str(path) for path in dataset_paths],
        "n_datasets_entrada": int(len(dataset_paths)),
        "model": model_name,
        "feature_mode": feature_mode,
        "model_params": model_kwargs,
        "split_strategy": "aleatorio_intrarun",
        "train_ratio": float(train_ratio),
        "max_seeds": None if max_seeds is None else int(max_seeds),
        "use_scaler": True,
        "scale_features": True,
        "scale_target": bool(escalar_y),
        "escalado": True,
    })

    metricas["random_state"] = int(random_state)

    if collect_sample_errors:
        metricas["sample_errors"] = sample_errors

    return metricas


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta un benchmark de surrogate sobre un dataset balanceado o "
            "directamente sobre datasets por seed."
        )
    )
    parser.add_argument(
        "--dataset",
        "--input",
        dest="dataset",
        default=None,
        help="Ruta al dataset HDF5 o Parquet.",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help="Lista de datasets por seed a evaluar sin pasar por dataset_completo.",
    )
    parser.add_argument(
        "--experiment-dir",
        default=None,
        help="Raiz del experimento para localizar automaticamente los datasets originales.",
    )
    parser.add_argument(
        "--algoritmo",
        choices=ALGORITMOS_MH,
        default=None,
        help="Algoritmo a usar junto con --experiment-dir.",
    )
    parser.add_argument(
        "--funcion",
        default=None,
        help="Funcion CEC concreta, por ejemplo f1, cuando se use --experiment-dir.",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["rbf", "svr", "mlp", "rsm", "random_forest", "hgb", "lasso", "xgboost"],
        help="Modelo surrogate a evaluar.",
    )
    parser.add_argument(
        "--feature-mode",
        required=True,
        choices=["x", "x_div"],
        help="Representacion de entrada a construir.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semilla para el split aleatorio. Default: 42.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Proporcion train dentro de la primera run para split aleatorio. Default: 0.7.",
    )
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=None,
        help="Limita la evaluacion a las primeras N seeds ordenadas. Default: usar todas.",
    )
    parser.add_argument(
        "--model-params-json",
        default=None,
        help="Ruta opcional a un JSON con los parametros del modelo.",
    )
    parser.add_argument(
        "--errors-out",
        default=None,
        help=(
            "Ruta opcional al artefacto de errores por muestra. "
            "Si termina en .csv se guarda CSV; si termina en .json se guarda JSON."
        ),
    )
    parser.add_argument(
        "--no-prediction-plots",
        action="store_true",
        help="Desactiva la generacion automatica de plots_predicciones.",
    )
    parser.add_argument(
        "--no-benchmark-summary",
        action="store_true",
        help="Desactiva la regeneracion automatica de resumen_benchmark_por_funcion.py.",
    )
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def resolver_datasets_entrada(args):
    fuentes = [
        args.dataset is not None,
        bool(args.inputs),
        args.experiment_dir is not None,
    ]
    if sum(fuentes) != 1:
        raise ValueError("Usa exactamente una fuente de entrada: --dataset, --inputs o --experiment-dir.")

    if args.dataset is not None:
        return [resolver_archivo_existente(args.dataset, arg_name="dataset")]

    if args.inputs:
        return [resolver_archivo_existente(path, arg_name="inputs") for path in args.inputs]

    if args.algoritmo is None:
        raise ValueError("Debes indicar --algoritmo cuando uses --experiment-dir.")

    return resolver_inputs_experimento(
        args.experiment_dir,
        args.algoritmo,
        funcion=args.funcion,
    )


def cargar_model_kwargs(ruta_json):
    if ruta_json is None:
        return {}
    ruta = resolver_archivo_existente(ruta_json, arg_name="model_params_json")
    with ruta.open("r", encoding="utf-8") as fh:
        return json.load(fh)

def construir_payload_metricas(metricas):
    payload = dict(metricas)
    payload.pop("sample_errors", None)
    return payload


def construir_payload_errores(metricas):
    return {
        "dataset": metricas.get("dataset"),
        "datasets": metricas.get("datasets"),
        "model": metricas["model"],
        "feature_mode": metricas["feature_mode"],
        "model_params": metricas["model_params"],
        "split_strategy": metricas["split_strategy"],
        "train_ratio": metricas.get("train_ratio"),
        "max_seeds": metricas.get("max_seeds"),
        "use_scaler": metricas["use_scaler"],
        "scale_features": metricas["scale_features"],
        "scale_target": metricas["scale_target"],
        "n_runs_evaluadas": metricas.get("n_runs_evaluadas"),
        "n_train": metricas.get("n_train"),
        "n_test": metricas.get("n_test"),
        "sample_errors": metricas.get("sample_errors", []),
    }


def generar_plots_predicciones(ruta_errores_json, model_name, split_strategy, random_state=None):
    ruta_errores_json = Path(ruta_errores_json)
    if ruta_errores_json.suffix.lower() != ".json":
        return

    script = (
        Path(__file__).resolve().parents[1]
        / "preprocesado_de_datos"
        / "utils_plots"
        / "plots_predicciones.py"
    )
    outdir = ruta_errores_json.parent / "plots_predicciones"

    titulo = model_name.upper()
    if random_state is not None:
        titulo = f"{titulo} seed{random_state}"
    else:
        titulo = f"{titulo} {split_strategy}"

    subprocess.run(
        [
            sys.executable,
            str(script),
            str(ruta_errores_json),
            "--outdir",
            str(outdir),
            "--title",
            titulo,
        ],
        check=True,
    )


def generar_resumen_benchmark_por_funcion(benchmark_dir):
    script = Path(__file__).resolve().parent / "resumen_benchmark_por_funcion.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--benchmark-dir",
            str(benchmark_dir),
        ],
        check=True,
    )


def imprimir_resumen(metricas):
    if metricas.get("dataset") is not None:
        print(f"dataset: {metricas['dataset']}")
    else:
        print(f"datasets: {metricas['n_datasets_entrada']}")
    print(f"model: {metricas['model']}")
    print(f"feature_mode: {metricas['feature_mode']}")
    print(f"split_strategy: {metricas['split_strategy']}")
    print(f"train_ratio: {metricas['train_ratio']}")
    print(f"use_scaler: {metricas['use_scaler']}")
    print(f"scale_features: {metricas['scale_features']}")
    print(f"scale_target: {metricas['scale_target']}")
    print(f"n_train: {metricas['n_train']}")
    print(f"n_test: {metricas['n_test']}")
    print(f"n_runs_evaluadas: {metricas['n_runs_evaluadas']}")
    print(f"MAE: {metricas['mae']:.6f} +- {metricas['mae_std']:.6f}")
    print(f"RMSE: {metricas['rmse']:.6f} +- {metricas['rmse_std']:.6f}")
    print(f"Spearman: {metricas['spearman']:.6f} +- {metricas['spearman_std']:.6f}")
    if "spearman_n_validas" in metricas and "spearman_n_nan" in metricas:
        print(
            f"Spearman runs validas: {metricas['spearman_n_validas']} "
            f"(NaN: {metricas['spearman_n_nan']})"
        )
    print(
        f"max_abs_error: {metricas['max_abs_error']:.6f} +- {metricas['max_abs_error_std']:.6f}"
    )
    print(
        f"max_pct_error: {metricas['max_pct_error']:.6f} +- {metricas['max_pct_error_std']:.6f}"
    )
    print(
        f"train_time_s: {metricas['train_time_s']:.6f} +- {metricas['train_time_s_std']:.6f}"
    )
    print(
        f"predict_time_s: {metricas['predict_time_s']:.6f} +- {metricas['predict_time_s_std']:.6f}"
    )


def guardar_errores_por_muestra_csv(errores, ruta_csv):
    ruta_csv = Path(ruta_csv)
    ruta_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for error in errores:
        for key in error.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["seed", "y_true", "y_pred", "error_abs", "error_pct"]
    with ruta_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(errores)


if __name__ == "__main__":
    args = parse_args()

    dataset_paths = ordenar_paths_por_seed(resolver_datasets_entrada(args))
    if len(dataset_paths) > 1 and args.max_seeds is not None:
        dataset_paths = dataset_paths[:args.max_seeds]

    model_dir = inferir_directorio_modelo(dataset_paths[0], args.model) if len(dataset_paths) == 1 else None
    ruta_metricas = Path(args.out) if args.out else None
    ruta_errores = Path(args.errors_out) if args.errors_out else None

    if ruta_metricas is None and model_dir is not None:
        ruta_metricas = model_dir / f"{args.model}_metricas.json"
    if ruta_errores is None and model_dir is not None:
        ruta_errores = model_dir / f"{args.model}_errores.json"

    metricas = ejecutar_benchmark(
        dataset_paths=dataset_paths,
        model_name=args.model,
        feature_mode=args.feature_mode,
        model_kwargs=cargar_model_kwargs(args.model_params_json),
        random_state=args.random_state,
        train_ratio=args.train_ratio,
        max_seeds=args.max_seeds if len(dataset_paths) == 1 else None,
        collect_sample_errors=(ruta_errores is not None),
    )

    payload_metricas = construir_payload_metricas(metricas)
    payload_errores = construir_payload_errores(metricas)

    if ruta_errores is not None:
        if ruta_errores.suffix.lower() == ".csv":
            guardar_errores_por_muestra_csv(metricas["sample_errors"], ruta_errores)
        else:
            escribir_json(ruta_errores, payload_errores)
            if not args.no_prediction_plots:
                generar_plots_predicciones(
                    ruta_errores,
                    model_name=args.model,
                    split_strategy=metricas["split_strategy"],
                    random_state=args.random_state,
                )

    if ruta_metricas is not None:
        escribir_json(ruta_metricas, payload_metricas)

    benchmark_dir = inferir_benchmark_dir_desde_candidatos(
        ruta_metricas,
        ruta_errores,
        model_dir,
        dataset_paths[0],
    )
    if benchmark_dir is not None and not args.no_benchmark_summary:
        generar_resumen_benchmark_por_funcion(benchmark_dir)

    imprimir_resumen(metricas)
