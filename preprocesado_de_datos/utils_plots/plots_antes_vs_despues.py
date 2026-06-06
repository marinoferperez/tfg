"""
plots_antes_vs_despues.py – Genera plots comparativos usando dataset_balanceado.metadata.json.

El objetivo es visualizar, de forma rápida y consistente, el efecto del
balanceo a nivel de experimento.

Por cada metadata detectada se generan:

1. comparacion de muestras antes vs despues,
2. distribucion por bins antes vs despues,
3. distribucion por seed antes vs despues,
4. fitness vs diversidad antes vs despues,
5. fitness vs generacion antes vs despues.

Los plots se guardan en el mismo directorio que contiene la metadata del
dataset balanceado. Por ejemplo:
    <experimento>/benchmark_surrogates/preprocesado/age/
    <experimento>/benchmark_surrogates/preprocesado/de/
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

try:
    from preprocesado_de_datos.utils.path_utils import (
        inferir_algoritmo_desde_artefacto,
        inferir_benchmark_dir,
        inferir_paths_runs_originales_desde_benchmark,
        leer_json,
        resolver_metadatas_balanceado,
        resolver_paths_cli,
    )
    from preprocesado_de_datos.utils.utils import cargar_dataset, concatenar_runs
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from preprocesado_de_datos.utils.path_utils import (
        inferir_algoritmo_desde_artefacto,
        inferir_benchmark_dir,
        inferir_paths_runs_originales_desde_benchmark,
        leer_json,
        resolver_metadatas_balanceado,
        resolver_paths_cli,
    )
    from preprocesado_de_datos.utils.utils import cargar_dataset, concatenar_runs

COLOR_ORIGINAL = "#4e79a7"
COLOR_BALANCEADO = "#f28e2b"
COLOR_AUX = "#59a14f"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Genera plots comparativos a partir de dataset_balanceado.metadata.json "
            "y los guarda junto a esa metadata."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=None,
        help=(
            "Rutas opcionales. Cada ruta puede ser un dataset_balanceado.metadata.json, "
            "una carpeta benchmark_surrogates o una carpeta superior que la contenga. "
            "Si no se indica nada, se escanea recursivamente desde el cwd."
        ),
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=None,
        help="Alias opcional de las rutas de entrada.",
    )
    return parser.parse_args()

def formatear_entero(valor):
    return f"{int(valor):,}".replace(",", ".")


def slug_fuente(path_metadata):
    parent_name = path_metadata.parent.name.lower()
    if parent_name in {"de", "age"}:
        return parent_name
    return path_metadata.stem.replace(".metadata", "")

def inferir_benchmark_dir_desde_metadata(path_metadata: Path) -> Path:
    benchmark_dir = inferir_benchmark_dir(path_metadata)
    if benchmark_dir is None:
        raise ValueError(f"La metadata no cuelga de una carpeta benchmark_surrogates: {path_metadata}")
    return benchmark_dir


def output_dir_desde_metadata(path_metadata):
    inferir_benchmark_dir_desde_metadata(path_metadata)
    outdir = Path(path_metadata).resolve().parent
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def estilo_ejes(ax):
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def detectar_clave_diversidad(dataset_original, dataset_balanceado):
    candidatas = ("div_dist_euclidea",)
    for clave in candidatas:
        if clave in dataset_original and clave in dataset_balanceado:
            return clave
    return None


def muestrear_indices(n_total, max_muestras, random_state):
    if n_total <= max_muestras:
        return np.arange(n_total, dtype=np.int64)
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(n_total, size=max_muestras, replace=False))


def inferir_paths_runs_originales(path_metadata):
    benchmark_dir = inferir_benchmark_dir_desde_metadata(path_metadata)
    fuente = inferir_algoritmo_desde_artefacto(path_metadata)
    if fuente not in {"de", "age"}:
        return []
    return inferir_paths_runs_originales_desde_benchmark(benchmark_dir, fuente)


def cargar_datasets_comparacion(path_metadata, metadata):
    path_balanceado = metadata.get("artefactos", {}).get("dataset_npz")
    candidatos_balanceado = []
    if path_balanceado:
        candidatos_balanceado.append(Path(path_balanceado))
    candidatos_balanceado.append(Path(path_metadata).with_suffix("").with_suffix(".npz"))

    path_balanceado_resuelto = None
    for candidato in candidatos_balanceado:
        if candidato.is_file():
            path_balanceado_resuelto = candidato
            break
    if path_balanceado_resuelto is None:
        return None, None

    paths_runs_originales = inferir_paths_runs_originales(path_metadata)
    if not paths_runs_originales:
        return None, None

    dataset_original = concatenar_runs(paths_runs_originales)
    dataset_balanceado = cargar_dataset(path_balanceado_resuelto)
    return dataset_original, dataset_balanceado


def plot_muestras(metadata, outpath, titulo_fuente):
    muestras = metadata["muestras"]
    original = int(muestras["original"])
    balanceado = int(muestras["balanceado"])
    retencion = float(muestras.get("retencion_pct", 100.0 * balanceado / max(original, 1)))

    labels = ["Original", "Balanceado"]
    valores = [original, balanceado]
    colores = [COLOR_ORIGINAL, COLOR_BALANCEADO]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    barras = ax.bar(labels, valores, color=colores, width=0.62, edgecolor="white", linewidth=1.0)

    ymax = max(valores) * 1.18 if max(valores) > 0 else 1.0
    ax.set_ylim(0, ymax)
    ax.set_ylabel("Numero de muestras")
    ax.set_title(f"Comparacion de muestras antes vs despues ({titulo_fuente})")
    estilo_ejes(ax)

    for barra, valor in zip(barras, valores):
        ax.text(
            barra.get_x() + barra.get_width() / 2.0,
            barra.get_height() + ymax * 0.015,
            formatear_entero(valor),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.text(
        0.98,
        0.96,
        f"Retencion: {retencion:.3f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f5f5f5", "edgecolor": "#d0d0d0"},
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_bins(metadata, outpath, titulo_fuente):
    antes = np.asarray(metadata["distribucion_bins"]["antes"], dtype=int)
    despues = np.asarray(metadata["distribucion_bins"]["despues"], dtype=int)
    n_bins = len(antes)
    x = np.arange(n_bins, dtype=float)
    ancho = 0.42

    max_por_bin = metadata.get("parametros", {}).get("max_por_bin")
    labels = [f"B{i+1}" for i in range(n_bins)]

    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.bar(
        x - ancho / 2.0,
        antes,
        width=ancho,
        color=COLOR_ORIGINAL,
        alpha=0.88,
        label="Antes",
    )
    ax.bar(
        x + ancho / 2.0,
        despues,
        width=ancho,
        color=COLOR_BALANCEADO,
        alpha=0.92,
        label="Despues",
    )

    ratio = float(np.max(antes) / max(np.min(antes[antes > 0]), 1)) if np.any(antes > 0) else 1.0
    if ratio > 30.0 or np.max(antes) > 20 * max(np.max(despues), 1):
        ax.set_yscale("log")
        ylabel = "Numero de muestras (escala log)"
    else:
        ylabel = "Numero de muestras"

    if max_por_bin is not None:
        ax.axhline(
            y=float(max_por_bin),
            color=COLOR_AUX,
            linestyle="--",
            linewidth=1.4,
            label=f"max_por_bin={max_por_bin}",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Bin de fitness")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Distribucion por bins antes vs despues ({titulo_fuente})")
    estilo_ejes(ax)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_seeds(metadata, outpath, titulo_fuente):
    dist_antes = metadata["distribucion_seeds"]["antes"]
    dist_despues = metadata["distribucion_seeds"]["despues"]

    seeds = np.array(sorted({int(k) for k in dist_antes} | {int(k) for k in dist_despues}), dtype=int)
    antes = np.asarray([int(dist_antes.get(str(seed), 0)) for seed in seeds], dtype=int)
    despues = np.asarray([int(dist_despues.get(str(seed), 0)) for seed in seeds], dtype=int)
    retencion = np.divide(
        despues,
        np.maximum(antes, 1),
        out=np.zeros_like(despues, dtype=float),
        where=antes > 0,
    ) * 100.0

    x = np.arange(len(seeds), dtype=float)
    ancho = 0.42

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(13, 7.8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.1]},
    )

    ax_top.bar(
        x - ancho / 2.0,
        antes,
        width=ancho,
        color=COLOR_ORIGINAL,
        alpha=0.85,
        label="Antes",
    )
    ax_top.bar(
        x + ancho / 2.0,
        despues,
        width=ancho,
        color=COLOR_BALANCEADO,
        alpha=0.92,
        label="Despues",
    )
    ax_top.set_yscale("log")
    ax_top.set_ylabel("Muestras por seed\n(escala log)")
    ax_top.set_title(f"Distribucion por seed antes vs despues ({titulo_fuente})")
    estilo_ejes(ax_top)
    ax_top.legend(loc="upper right")

    ax_bottom.bar(
        x,
        retencion,
        width=0.72,
        color=COLOR_AUX,
        alpha=0.88,
        edgecolor="white",
        linewidth=0.8,
    )
    ax_bottom.axhline(
        y=float(np.mean(retencion)),
        color="#222222",
        linestyle="--",
        linewidth=1.1,
        label=f"media={np.mean(retencion):.3f}%",
    )
    ax_bottom.set_ylabel("Retencion %")
    ax_bottom.set_xlabel("Seed")
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels([str(seed) for seed in seeds], rotation=90 if len(seeds) > 20 else 0)
    estilo_ejes(ax_bottom)
    ax_bottom.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_fitness_vs_diversidad_comparativo(
    dataset_original,
    dataset_balanceado,
    outpath,
    max_muestras=25000,
    random_state=42,
):
    clave_div = detectar_clave_diversidad(dataset_original, dataset_balanceado)
    if clave_div is None:
        return None

    fit_orig = np.asarray(dataset_original["fitness"], dtype=float)
    fit_bal = np.asarray(dataset_balanceado["fitness"], dtype=float)
    div_orig = np.asarray(dataset_original[clave_div], dtype=float)
    div_bal = np.asarray(dataset_balanceado[clave_div], dtype=float)

    x_low = min(
        float(np.percentile(div_orig, 0.5)),
        float(np.percentile(div_bal, 0.5)),
    )
    x_high = max(
        float(np.percentile(div_orig, 99.5)),
        float(np.percentile(div_bal, 99.5)),
    )
    y_low = min(
        float(np.percentile(fit_orig, 0.5)),
        float(np.percentile(fit_bal, 0.5)),
    )
    y_high = max(
        float(np.percentile(fit_orig, 99.5)),
        float(np.percentile(fit_bal, 99.5)),
    )

    idx_orig = muestrear_indices(len(fit_orig), max_muestras=max_muestras, random_state=random_state)
    idx_bal = muestrear_indices(len(fit_bal), max_muestras=max_muestras, random_state=random_state)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)
    paneles = (
        ("Original", div_orig[idx_orig], fit_orig[idx_orig], COLOR_ORIGINAL),
        ("Balanceado", div_bal[idx_bal], fit_bal[idx_bal], COLOR_BALANCEADO),
    )

    for ax, (titulo, x, y, color) in zip(axes, paneles):
        ax.scatter(
            x,
            y,
            s=8,
            alpha=0.18,
            c=color,
            edgecolors="none",
            rasterized=True,
        )
        ax.set_title(f"{titulo} (n={len(x)})")
        ax.grid(alpha=0.20)

    axes[0].set_ylabel("Fitness")
    for ax in axes:
        ax.set_xlabel(clave_div)
        ax.set_xlim(x_low, x_high)
        ax.set_ylim(y_low, y_high)

    fig.suptitle("Fitness vs diversidad", y=0.98)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)
    return clave_div


def plot_fitness_vs_generacion_comparativo(
    dataset_original,
    dataset_balanceado,
    outpath,
    max_muestras=25000,
    random_state=42,
):
    fit_orig = np.asarray(dataset_original["fitness"], dtype=float)
    fit_bal = np.asarray(dataset_balanceado["fitness"], dtype=float)
    gen_orig = np.asarray(dataset_original["generacion"], dtype=float)
    gen_bal = np.asarray(dataset_balanceado["generacion"], dtype=float)

    x_low = min(
        float(np.percentile(gen_orig, 0.5)),
        float(np.percentile(gen_bal, 0.5)),
    )
    x_high = max(
        float(np.percentile(gen_orig, 99.5)),
        float(np.percentile(gen_bal, 99.5)),
    )
    y_low = min(
        float(np.percentile(fit_orig, 0.5)),
        float(np.percentile(fit_bal, 0.5)),
    )
    y_high = max(
        float(np.percentile(fit_orig, 99.5)),
        float(np.percentile(fit_bal, 99.5)),
    )

    idx_orig = muestrear_indices(len(fit_orig), max_muestras=max_muestras, random_state=random_state)
    idx_bal = muestrear_indices(len(fit_bal), max_muestras=max_muestras, random_state=random_state)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)
    paneles = (
        ("Original", gen_orig[idx_orig], fit_orig[idx_orig], COLOR_ORIGINAL),
        ("Balanceado", gen_bal[idx_bal], fit_bal[idx_bal], COLOR_BALANCEADO),
    )

    for ax, (titulo, x, y, color) in zip(axes, paneles):
        ax.scatter(
            x,
            y,
            s=8,
            alpha=0.18,
            c=color,
            edgecolors="none",
            rasterized=True,
        )
        ax.set_title(f"{titulo} (n={len(x)})")
        ax.grid(alpha=0.20)

    axes[0].set_ylabel("Fitness")
    for ax in axes:
        ax.set_xlabel("generacion")
        ax.set_xlim(x_low, x_high)
        ax.set_ylim(y_low, y_high)

    fig.suptitle("Fitness vs generacion", y=0.98)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def generar_plots_para_metadata(path_metadata):
    metadata = leer_json(path_metadata)
    outdir = output_dir_desde_metadata(path_metadata)
    fuente = slug_fuente(path_metadata)
    dataset_original, dataset_balanceado = cargar_datasets_comparacion(path_metadata, metadata)

    plot_muestras(
        metadata,
        outdir / f"{fuente}_muestras_antes_vs_despues.png",
        titulo_fuente=fuente.upper(),
    )
    plot_bins(
        metadata,
        outdir / f"{fuente}_distribucion_bins_antes_vs_despues.png",
        titulo_fuente=fuente.upper(),
    )
    plot_seeds(
        metadata,
        outdir / f"{fuente}_distribucion_seeds_antes_vs_despues.png",
        titulo_fuente=fuente.upper(),
    )

    if dataset_original is not None and dataset_balanceado is not None:
        plot_fitness_vs_diversidad_comparativo(
            dataset_original,
            dataset_balanceado,
            outdir / f"{fuente}_fitness_vs_diversidad_antes_vs_despues.png",
        )
        plot_fitness_vs_generacion_comparativo(
            dataset_original,
            dataset_balanceado,
            outdir / f"{fuente}_fitness_vs_generacion_antes_vs_despues.png",
        )

    return outdir


def main():
    args = parse_args()
    raw_paths = resolver_paths_cli(inputs_opt=args.inputs, inputs_pos=args.paths, allow_empty=True)
    metadata_paths = resolver_metadatas_balanceado(raw_paths)
    if not metadata_paths:
        raise FileNotFoundError("No se encontraron archivos de metadata para dataset_balanceado.")

    print(f"Metadatas detectadas: {len(metadata_paths)}")
    for path_metadata in metadata_paths:
        outdir = generar_plots_para_metadata(path_metadata)
        print(f"[ok] {path_metadata}")
        print(f"     plots -> {outdir}")


if __name__ == "__main__":
    main()
