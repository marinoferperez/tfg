#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

CACHE_ROOT = Path(tempfile.gettempdir()) / "tfg_matplotlib_cache"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("results/cec/cec2017_d10_tam50_reinicio_seleccionado")
OUTDIR = Path("memoria/figuras/surrogates_offline")
OUTDIR.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = {
    "RSM": {
        "root": ROOT / "benchmarking_surrogates_final_full/future_next/rsm_degree_2/rsm/no_acumulativo",
        "model": "rsm",
    },
    "RBF": {
        "root": ROOT / "benchmarking_surrogates_final_full/future_next/rbf_neighbors_50/rbf/no_acumulativo",
        "model": "rbf",
    },
}


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 11,
        "axes.labelsize": 10.5,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 120,
    }
)


def cargar_metricas_resumen():
    filas = []

    for modelo, cfg in EXPERIMENTS.items():
        root = cfg["root"]
        model = cfg["model"]

        if not root.exists():
            raise FileNotFoundError(f"No existe el directorio esperado: {root}")

        for path in root.rglob(f"{model}_metricas.json"):
            rel = path.relative_to(root).parts

            # Resumen raíz por función y metaheurística:
            # fX / algoritmo / modelo / modelo_metricas.json
            # Se excluyen los JSON internos de cada bloque temporal.
            if len(rel) != 4:
                continue

            funcion, algoritmo, _, _ = rel

            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)

            filas.append(
                {
                    "modelo": modelo,
                    "funcion": funcion,
                    "funcid": int(funcion[1:]),
                    "algoritmo": algoritmo.upper(),
                    "spearman": float(data["spearman"]),
                    "nrmse": float(data["nrmse"]),
                    "nmae": float(data["nmae"]),
                    "train_time_s": float(data["train_time_s"]),
                    "predict_time_s": float(data["predict_time_s"]),
                }
            )

    return filas


def media_por_funcion(filas, metrica):
    modelos = list(EXPERIMENTS.keys())
    funciones = list(range(1, 31))
    matriz = np.full((len(funciones), len(modelos)), np.nan)

    for i, funcid in enumerate(funciones):
        for j, modelo in enumerate(modelos):
            valores = [
                fila[metrica]
                for fila in filas
                if fila["modelo"] == modelo and fila["funcid"] == funcid
            ]
            if valores:
                matriz[i, j] = float(np.mean(valores))

    return funciones, modelos, matriz


def generar_heatmap_nrmse(filas):
    funciones, modelos, matriz = media_por_funcion(filas, "nrmse")

    # Escala logarítmica para que f2 no oculte el resto de funciones.
    matriz_log = np.log10(np.maximum(matriz, 1e-12))

    fig, ax = plt.subplots(figsize=(6.2, 8.8))
    im = ax.imshow(matriz_log, aspect="auto", cmap="viridis")

    ax.set_xticks(np.arange(len(modelos)))
    ax.set_xticklabels(modelos)
    ax.set_yticks(np.arange(len(funciones)))
    ax.set_yticklabels([f"$f_{{{i}}}$" for i in funciones])

    ax.set_xlabel("Modelo")
    ax.set_ylabel("Función CEC2017")
    ax.set_title(r"nRMSE medio por función ($\log_{10}$)")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\log_{10}(\mathrm{nRMSE})$")

    ax.set_xticks(np.arange(-0.5, len(modelos), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(funciones), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Anotación numérica de cada celda en la misma escala del color.
    umbral_color = np.nanmean(matriz_log)
    for i in range(len(funciones)):
        for j in range(len(modelos)):
            valor = matriz_log[i, j]
            if np.isnan(valor):
                continue

            color_texto = "white" if valor > umbral_color else "#222222"
            ax.text(
                j,
                i,
                f"{valor:.1f}",
                ha="center",
                va="center",
                fontsize=7.2,
                color=color_texto,
            )

    fig.tight_layout()

    png = OUTDIR / "heatmap_nrmse_final_cec2017.png"
    pdf = OUTDIR / "heatmap_nrmse_final_cec2017.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Generado: {png}")
    print(f"Generado: {pdf}")


def generar_heatmap_spearman(filas):
    funciones, modelos, matriz = media_por_funcion(filas, "spearman")

    fig, ax = plt.subplots(figsize=(6.2, 8.8))
    im = ax.imshow(matriz, aspect="auto", cmap="RdYlGn", vmin=-1.0, vmax=1.0)

    ax.set_xticks(np.arange(len(modelos)))
    ax.set_xticklabels(modelos)
    ax.set_yticks(np.arange(len(funciones)))
    ax.set_yticklabels([f"$f_{{{i}}}$" for i in funciones])

    ax.set_xlabel("Modelo")
    ax.set_ylabel("Función CEC2017")
    ax.set_title("Spearman medio por función")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman")

    ax.set_xticks(np.arange(-0.5, len(modelos), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(funciones), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(len(funciones)):
        for j in range(len(modelos)):
            valor = matriz[i, j]
            if np.isnan(valor):
                continue

            color_texto = "white" if abs(valor) > 0.55 else "#222222"
            ax.text(
                j,
                i,
                f"{valor:.2f}",
                ha="center",
                va="center",
                fontsize=7.2,
                color=color_texto,
            )

    fig.tight_layout()

    png = OUTDIR / "heatmap_spearman_final_cec2017.png"
    pdf = OUTDIR / "heatmap_spearman_final_cec2017.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Generado: {png}")
    print(f"Generado: {pdf}")


def estilizar_boxplot(bp, colores):
    for patch, color in zip(bp["boxes"], colores):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor("#333333")
        patch.set_linewidth(1.0)

    for median in bp["medians"]:
        median.set_color("#111111")
        median.set_linewidth(1.3)

    for whisker in bp["whiskers"]:
        whisker.set_color("#555555")
        whisker.set_linewidth(0.9)

    for cap in bp["caps"]:
        cap.set_color("#555555")
        cap.set_linewidth(0.9)

    for flier in bp["fliers"]:
        flier.set_marker("o")
        flier.set_markersize(3)
        flier.set_markerfacecolor("#555555")
        flier.set_markeredgecolor("#555555")
        flier.set_alpha(0.45)


def generar_boxplots_errores(filas):
    modelos = list(EXPERIMENTS.keys())
    metricas = [
        ("nrmse", "nRMSE"),
        ("nmae", "nMAE"),
    ]
    nombres_archivos = [
        ("boxplot_nrmse_final_cec2017", r"nRMSE", r"$\log_{10}$ de la métrica"),
        ("boxplot_nmae_final_cec2017", r"nMAE", r"$\log_{10}$ de la métrica"),
    ]
    colores = ["#4C78A8", "#F58518"]

    for (metrica, titulo), (nombre_archivo, _, ylabel) in zip(metricas, nombres_archivos):
        datos = []
        for modelo in modelos:
            valores = [
                fila[metrica]
                for fila in filas
                if fila["modelo"] == modelo
            ]
            datos.append(np.log10(np.maximum(valores, 1e-12)))

        fig, ax = plt.subplots(figsize=(5.0, 4.2))
        bp = ax.boxplot(
            datos,
            tick_labels=modelos,
            patch_artist=True,
            showmeans=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "#222222",
                "markeredgecolor": "#222222",
                "markersize": 4,
            },
        )
        estilizar_boxplot(bp, colores)

        ax.set_title(titulo)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.45)

        fig.tight_layout()
        png = OUTDIR / f"{nombre_archivo}.png"
        pdf = OUTDIR / f"{nombre_archivo}.pdf"
        fig.savefig(png, dpi=300, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)

        print(f"Generado: {png}")
        print(f"Generado: {pdf}")


def verificar_cobertura(filas):
    for modelo in EXPERIMENTS:
        funcs = sorted({fila["funcid"] for fila in filas if fila["modelo"] == modelo})
        algs = sorted({fila["algoritmo"] for fila in filas if fila["modelo"] == modelo})
        n = sum(1 for fila in filas if fila["modelo"] == modelo)

        print(f"{modelo}: {len(funcs)} funciones, algoritmos={algs}, resúmenes={n}")

        if funcs != list(range(1, 31)):
            raise ValueError(f"{modelo}: no están las 30 funciones CEC2017.")

        if algs != ["AGE", "DE", "SHADE"]:
            raise ValueError(f"{modelo}: no están AGE, DE y SHADE.")

        if n != 90:
            raise ValueError(
                f"{modelo}: se esperaban 90 resúmenes raíz "
                f"(30 funciones x 3 metaheurísticas), pero hay {n}."
            )


def main():
    filas = cargar_metricas_resumen()
    verificar_cobertura(filas)
    generar_heatmap_nrmse(filas)
    generar_heatmap_spearman(filas)
    generar_boxplots_errores(filas)


if __name__ == "__main__":
    main()
