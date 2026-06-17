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
METRICS_ROOT = (
    ROOT
    / "benchmarking_surrogates_offline_rbf_rsm_tuned"
    / "future_next"
    / "rbf"
    / "no_acumulativo"
)
OUTDIR = Path("memoria/figuras/surrogates_offline")
OUTDIR.mkdir(parents=True, exist_ok=True)

ALGOS = ["age", "de", "shade"]
LABELS = {"age": "AGE", "de": "DE", "shade": "SHADE"}
FUNCS = [f"f{i}" for i in range(1, 31)]


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 11,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 120,
    }
)


def cargar_matriz_spearman() -> np.ndarray:
    matriz = np.full((len(FUNCS), len(ALGOS)), np.nan)

    for i, func in enumerate(FUNCS):
        for j, algo in enumerate(ALGOS):
            path = METRICS_ROOT / func / algo / "rbf" / "rbf_metricas.json"
            if not path.exists():
                raise FileNotFoundError(f"No existe el fichero esperado: {path}")

            data = json.loads(path.read_text(encoding="utf-8"))
            runs = data.get("runs", [])
            seeds = {run.get("seed") for run in runs}
            bloques = {run.get("batch_train") for run in runs}
            if seeds != set(range(1, 52)):
                raise ValueError(
                    f"{path}: se esperaban las 51 semillas, pero hay {len(seeds)}."
                )
            if bloques != {1, 2, 3, 4}:
                raise ValueError(
                    f"{path}: se esperaban los 4 bloques de entrenamiento, "
                    f"pero aparecen {sorted(bloques)}."
                )
            if len(runs) != 51 * 4:
                raise ValueError(
                    f"{path}: se esperaban 204 registros internos "
                    f"(51 semillas x 4 bloques), pero hay {len(runs)}."
                )

            matriz[i, j] = float(data["spearman"])

    return matriz


def generar_heatmap(matriz: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 8.8))
    im = ax.imshow(matriz, aspect="auto", cmap="RdYlGn", vmin=-1.0, vmax=1.0)

    ax.set_xticks(np.arange(len(ALGOS)))
    ax.set_xticklabels([LABELS[a] for a in ALGOS])
    ax.set_yticks(np.arange(len(FUNCS)))
    ax.set_yticklabels([f"$f_{{{i}}}$" for i in range(1, 31)])

    ax.set_xlabel("Metaheurística")
    ax.set_ylabel("Función CEC2017")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman")

    ax.set_xticks(np.arange(-0.5, len(ALGOS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(FUNCS), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(len(FUNCS)):
        for j in range(len(ALGOS)):
            valor = matriz[i, j]
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

    png = OUTDIR / "heatmap_spearman_rbf_final_por_mh.png"
    pdf = OUTDIR / "heatmap_spearman_rbf_final_por_mh.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Generado: {png}")
    print(f"Generado: {pdf}")


def main() -> None:
    if not METRICS_ROOT.exists():
        raise FileNotFoundError(f"No existe el directorio esperado: {METRICS_ROOT}")

    matriz = cargar_matriz_spearman()
    print("Cobertura: 30 funciones x 3 metaheurísticas x 51 semillas x 4 bloques.")
    for j, algo in enumerate(ALGOS):
        print(f"{LABELS[algo]}: Spearman medio = {np.nanmean(matriz[:, j]):.4f}")
    generar_heatmap(matriz)


if __name__ == "__main__":
    main()
