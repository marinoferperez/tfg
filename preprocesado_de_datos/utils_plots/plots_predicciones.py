"""
plots_predicciones.py – Graficos para evaluar un surrogate como predictor puro.

La entrada esperada es un JSON de benchmark que contenga la clave
sample_errors, generado por benchmark_runner_cec.py al activar la recogida de
errores por muestra.

Por cada entrada genera un conjunto de plots diagnosticos pensados para
analizar calidad predictiva al 100%, no solo utilidad como ranking surrogate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig_codex"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocesado_de_datos.utils.path_utils import (
    asegurar_directorio,
    escribir_json,
    resolver_input_cli,
)


COLOR_MAIN = "#4e79a7"
COLOR_AUX = "#f28e2b"
COLOR_ERR = "#e15759"
COLOR_OK = "#59a14f"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Genera plots de diagnostico predictivo a partir de un JSON de "
            "benchmark con sample_errors."
        )
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=None,
        help="Ruta opcional al JSON del benchmark.",
    )
    parser.add_argument(
        "--input",
        dest="input_opt",
        default=None,
        help="Ruta al JSON del benchmark.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Directorio de salida. Por defecto se crea plots_predicciones junto a la entrada.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Titulo base opcional para los plots.",
    )
    return parser.parse_args()


def cargar_errores(path_input):
    path_input = resolver_input_cli(input_pos=path_input, arg_name="input")
    if path_input.suffix.lower() != ".json":
        raise ValueError("La entrada debe ser un .json")

    with path_input.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if "sample_errors" not in data:
        raise ValueError(
            "El JSON no contiene sample_errors. Ejecuta benchmark_runner_cec.py "
            "guardando errores por muestra."
        )
    df = pd.DataFrame(data["sample_errors"])

    columnas_obligatorias = {"y_true", "y_pred"}
    faltan = columnas_obligatorias - set(df.columns)
    if faltan:
        raise ValueError(f"Faltan columnas obligatorias en la entrada: {sorted(faltan)}")

    df = df.copy()
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    if "error_abs" in df.columns:
        df["error_abs"] = pd.to_numeric(df["error_abs"], errors="coerce")
    else:
        df["error_abs"] = (df["y_true"] - df["y_pred"]).abs()

    if "error_pct" in df.columns:
        df["error_pct"] = pd.to_numeric(df["error_pct"], errors="coerce")
    else:
        denom = df["y_true"].abs().replace(0, np.nan)
        df["error_pct"] = df["error_abs"] / denom * 100.0

    df["error_signed"] = df["y_pred"] - df["y_true"]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["y_true", "y_pred", "error_abs", "error_signed"])
    return df


def resolver_outdir(path_input, outdir):
    if outdir is not None:
        outdir = asegurar_directorio(outdir)
    else:
        outdir = asegurar_directorio(Path(path_input).parent / "plots_predicciones")
    return outdir


def titulo_base(path_input, title):
    if title:
        return title
    return Path(path_input).stem


def estilo_ejes(ax):
    ax.grid(alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_real_vs_predicho(df, outpath, title):
    y_true = df["y_true"].to_numpy(dtype=float)
    y_pred = df["y_pred"].to_numpy(dtype=float)

    low = min(np.min(y_true), np.min(y_pred))
    high = max(np.max(y_true), np.max(y_pred))

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.scatter(
        y_true,
        y_pred,
        s=14,
        alpha=0.28,
        color=COLOR_MAIN,
        edgecolors="none",
        rasterized=True,
    )
    ax.plot([low, high], [low, high], linestyle="--", color=COLOR_ERR, linewidth=1.5, label="Ideal: y=x")

    ax.set_title(f"Real vs predicho\n{title}")
    ax.set_xlabel("Fitness real")
    ax.set_ylabel("Fitness predicho")
    estilo_ejes(ax)
    ax.legend(loc="best")

    if low > 0 and high > 0:
        ax.set_xscale("log")
        ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_residual_vs_fitness(df, outpath, title):
    x = df["y_true"].to_numpy(dtype=float)
    residual = df["error_signed"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.scatter(
        x,
        residual,
        s=10,
        alpha=0.22,
        color=COLOR_MAIN,
        edgecolors="none",
        rasterized=True,
    )
    ax.axhline(0.0, linestyle="--", linewidth=1.3, color=COLOR_ERR)
    ax.set_title(f"Residual vs fitness real\n{title}")
    ax.set_xlabel("Fitness real")
    ax.set_ylabel("Residual (pred - real)")
    estilo_ejes(ax)

    if np.all(x > 0):
        ax.set_xscale("log")

    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_error_absoluto_vs_fitness(df, outpath, title):
    x = df["y_true"].to_numpy(dtype=float)
    err_abs = df["error_abs"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.scatter(
        x,
        err_abs,
        s=10,
        alpha=0.22,
        color=COLOR_ERR,
        edgecolors="none",
        rasterized=True,
    )
    ax.set_title(f"Error absoluto vs fitness real\n{title}")
    ax.set_xlabel("Fitness real")
    ax.set_ylabel("Error absoluto")
    estilo_ejes(ax)

    if np.all(x > 0):
        ax.set_xscale("log")
    if np.all(err_abs > 0):
        ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_ecdf_error_absoluto(df, outpath, title):
    err_abs = np.sort(df["error_abs"].to_numpy(dtype=float))
    y = np.arange(1, len(err_abs) + 1, dtype=float) / len(err_abs)

    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    ax.plot(err_abs, y, color=COLOR_OK, linewidth=2.0)
    ax.set_title(f"ECDF del error absoluto\n{title}")
    ax.set_xlabel("Error absoluto")
    ax.set_ylabel("Probabilidad acumulada")
    estilo_ejes(ax)
    if np.all(err_abs > 0):
        ax.set_xscale("log")

    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_curva_ordenada(df, outpath, title):
    orden = np.argsort(df["y_true"].to_numpy(dtype=float))
    y_true = df["y_true"].to_numpy(dtype=float)[orden]
    y_pred = df["y_pred"].to_numpy(dtype=float)[orden]
    x = np.arange(len(y_true), dtype=int)

    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    ax.plot(x, y_true, color=COLOR_ERR, linewidth=1.8, label="Real")
    ax.plot(x, y_pred, color=COLOR_MAIN, linewidth=1.4, alpha=0.92, label="Predicho")
    ax.set_title(f"Curva ordenada por fitness real\n{title}")
    ax.set_xlabel("Muestras ordenadas por y_true")
    ax.set_ylabel("Fitness")
    estilo_ejes(ax)
    ax.legend(loc="best")

    if np.all(y_true > 0) and np.all(y_pred > 0):
        ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def guardar_resumen(df, outpath):
    resumen = {
        "n_muestras": int(len(df)),
        "mae_empirico": float(df["error_abs"].mean()),
        "mediana_error_abs": float(df["error_abs"].median()),
        "rmse_empirico": float(np.sqrt(np.mean(np.square(df["error_signed"])))),
        "bias_medio": float(df["error_signed"].mean()),
        "error_abs_p90": float(df["error_abs"].quantile(0.90)),
        "error_abs_p95": float(df["error_abs"].quantile(0.95)),
        "error_abs_p99": float(df["error_abs"].quantile(0.99)),
    }
    escribir_json(outpath, resumen)


def main():
    args = parse_args()
    input_path = resolver_input_cli(input_opt=args.input_opt, input_pos=args.input_path, arg_name="input")
    df = cargar_errores(input_path)
    outdir = resolver_outdir(input_path, args.outdir)
    title = titulo_base(input_path, args.title)

    plot_real_vs_predicho(df, outdir / "real_vs_predicho.png", title)
    plot_residual_vs_fitness(df, outdir / "residual_vs_fitness_real.png", title)
    plot_error_absoluto_vs_fitness(df, outdir / "error_absoluto_vs_fitness_real.png", title)
    plot_ecdf_error_absoluto(df, outdir / "ecdf_error_absoluto.png", title)
    plot_curva_ordenada(df, outdir / "curva_ordenada_real_vs_predicha.png", title)
    guardar_resumen(df, outdir / "resumen_predicciones.json")

    print(f"n_muestras={len(df)}")
    print(f"plots -> {outdir}")


if __name__ == "__main__":
    main()
