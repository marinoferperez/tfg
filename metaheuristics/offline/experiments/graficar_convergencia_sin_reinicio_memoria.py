#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

import numpy as np

CACHE_ROOT = Path(tempfile.gettempdir()) / "tfg_matplotlib_cache"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "results" / "cec" / "cec2017_d10_tam50"
OUT_DIR = ROOT / "memoria" / "figuras" / "plots_sin_reinicio"
FUNCIONES = [1, 10, 29]
ALGORITMOS = ["age", "de", "shade"]
COLORES = {
    "age": "#4e79a7",
    "de": "#f28e2b",
    "shade": "#59a14f",
}
LABELS = {
    "age": "AGE",
    "de": "DE",
    "shade": "SHADE",
}


def cargar_curva(csv_path: Path, columna: str) -> tuple[np.ndarray, np.ndarray] | None:
    evaluaciones = []
    valores = []
    with csv_path.open("r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        for fila in reader:
            if not fila.get("evaluaciones") or not fila.get(columna):
                continue
            evaluaciones.append(int(float(fila["evaluaciones"])))
            valores.append(float(fila[columna]))

    if not evaluaciones:
        return None

    pares = sorted(zip(evaluaciones, valores), key=lambda p: p[0])
    x_limpio = []
    y_limpio = []
    for x, y in pares:
        if x_limpio and x == x_limpio[-1]:
            y_limpio[-1] = y
        else:
            x_limpio.append(x)
            y_limpio.append(y)

    x = np.asarray(x_limpio, dtype=int)
    y = np.asarray(y_limpio, dtype=float)
    if columna == "min/mejor_hasta_ahora":
        y = np.minimum.accumulate(y)
    return x, y


def alinear_curvas(curvas: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    malla = np.unique(np.concatenate([x for x, _ in curvas]))
    matriz = np.empty((len(curvas), len(malla)), dtype=float)

    for i, (x, y) in enumerate(curvas):
        idx = np.searchsorted(x, malla, side="right") - 1
        idx[idx < 0] = 0
        matriz[i, :] = y[idx]

    return malla, np.mean(matriz, axis=0)


def curvas_algoritmo(func_id: int, algoritmo: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base = RESULTS_DIR / f"f{func_id}" / "metricas_runs" / "cec2017" / algoritmo
    curvas_mejor = []
    curvas_media = []

    for csv_path in sorted(base.rglob("resultados_*.csv")):
        mejor = cargar_curva(csv_path, "min/mejor_hasta_ahora")
        media = cargar_curva(csv_path, "promedio")
        if mejor is None or media is None:
            continue
        curvas_mejor.append(mejor)
        curvas_media.append(media)

    if not curvas_mejor:
        raise FileNotFoundError(f"No se encontraron curvas para {base}")

    x_mejor, y_mejor = alinear_curvas(curvas_mejor)
    x_media, y_media = alinear_curvas(curvas_media)
    optimo = 100.0 * func_id
    y_mejor = np.log10(np.maximum(y_mejor - optimo, 0.0) + 1.0)
    y_media = np.log10(np.maximum(y_media - optimo, 0.0) + 1.0)
    return x_mejor, y_mejor, x_media, y_media


def generar_figura(func_id: int) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.5))

    for algoritmo in ALGORITMOS:
        x_mejor, y_mejor, x_media, y_media = curvas_algoritmo(func_id, algoritmo)
        color = COLORES[algoritmo]
        ax.step(x_mejor, y_mejor, where="post", color=color, linewidth=1.8)
        ax.step(x_media, y_media, where="post", color=color, linestyle="--", linewidth=1.3)

    ax.set_xlabel("Evaluaciones", fontsize=10)
    ax.set_ylabel("")
    ax.grid(alpha=0.22)
    ax.tick_params(labelsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0, y=0.03)
    fig.tight_layout(pad=0.45)
    fig.savefig(OUT_DIR / f"curva_convergencia_cec2017_f{func_id}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"curva_convergencia_cec2017_f{func_id}.pdf", bbox_inches="tight")
    plt.close(fig)


def generar_leyenda() -> None:
    handles = []
    labels = []
    for algoritmo in ALGORITMOS:
        handles.append(Line2D([0], [0], color=COLORES[algoritmo], linewidth=1.8))
        labels.append(f"{LABELS[algoritmo]} mejor")
        handles.append(Line2D([0], [0], color=COLORES[algoritmo], linestyle="--", linewidth=1.3))
        labels.append(f"{LABELS[algoritmo]} media")

    fig, ax = plt.subplots(figsize=(7.2, 1.35))
    ax.axis("off")
    ax.legend(
        handles,
        labels,
        loc="center",
        ncol=3,
        frameon=False,
        fontsize=18,
        handlelength=2.4,
        columnspacing=1.5,
        labelspacing=0.7,
    )
    fig.savefig(OUT_DIR / "leyenda_convergencia_cec2017.png", dpi=300, bbox_inches="tight", transparent=True)
    fig.savefig(OUT_DIR / "leyenda_convergencia_cec2017.pdf", bbox_inches="tight", transparent=True)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for func_id in FUNCIONES:
        generar_figura(func_id)
    generar_leyenda()


if __name__ == "__main__":
    main()
