#!/usr/bin/env python3
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_ROOT = Path(tempfile.gettempdir()) / "tfg_matplotlib_cache"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = ROOT / "results" / "cec"
OUT_PATH = ROOT / "memoria" / "figuras" / "rendimiento_por_ventana_estancamiento.png"

VARIANTES = [
    ("Sin", RESULTS_ROOT / "cec2017_d10_tam50"),
    ("1%", RESULTS_ROOT / "cec2017_d10_tam50_reinicio_pat001"),
    ("3%", RESULTS_ROOT / "cec2017_d10_tam50_reinicio_pat003"),
    ("5%", RESULTS_ROOT / "cec2017_d10_tam50_reinicio_pat005"),
    ("7%", RESULTS_ROOT / "cec2017_d10_tam50_reinicio_pat007"),
    ("10%", RESULTS_ROOT / "cec2017_d10_tam50_reinicio_pat01"),
]

ALGORITMOS = ["age", "de", "shade"]
LABELS = {"age": "AGE", "de": "DE", "shade": "SHADE"}
COLORES = {"age": "#4e79a7", "de": "#f28e2b", "shade": "#59a14f"}
MARCADORES = {"age": "o", "de": "s", "shade": "^"}


def cargar_valores(base: Path, algoritmo: str) -> np.ndarray:
    valores = []
    for runs_csv in sorted(base.glob("f*/runs.csv")):
        df = pd.read_csv(runs_csv)
        df = df[df["algoritmo"] == algoritmo]
        if df.empty:
            continue
        error = df["cec_error"].astype(float).to_numpy()
        valores.extend(np.log10(np.maximum(error, 0.0) + 1.0))

    if not valores:
        raise FileNotFoundError(f"No hay datos para {algoritmo} en {base}")
    return np.asarray(valores, dtype=float)


def main() -> None:
    x = np.arange(len(VARIANTES), dtype=float)
    etiquetas_x = [label for label, _ in VARIANTES]

    fig, ax = plt.subplots(figsize=(9.2, 4.8))

    for algoritmo in ALGORITMOS:
        medias = [
            float(np.mean(cargar_valores(base, algoritmo)))
            for _, base in VARIANTES
        ]
        ax.plot(
            x,
            medias,
            marker=MARCADORES[algoritmo],
            markersize=7,
            linewidth=2.2,
            color=COLORES[algoritmo],
            label=LABELS[algoritmo],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas_x)
    ax.set_xlabel("Ventana de estancamiento (% del presupuesto)", fontsize=11)
    ax.set_ylabel("Error medio log.", fontsize=11)
    ax.grid(alpha=0.28)
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="center right", fontsize=11, frameon=True)
    fig.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
