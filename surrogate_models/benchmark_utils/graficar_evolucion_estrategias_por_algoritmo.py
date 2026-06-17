#!/usr/bin/env python3
"""
Genera la Figura 7.7 rediseñada: evolución del Spearman medio por bloque
temporal para las estrategias acumulativa y no acumulativa, con un panel
por algoritmo (AGE, DE, SHADE).

Salida: memoria/figuras/surrogates_offline/evolucion_spearman_por_bloque_por_algoritmo.{png,pdf}
"""
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
import pandas as pd

BASE_DIR = Path(
    "results/cec/cec2017_d10_tam50_reinicio_seleccionado"
    "/benchmarking_surrogates_offline_next/future_next"
)
OUT_DIR = Path("memoria/figuras/surrogates_offline")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELOS = ["hgb", "lasso", "mlp", "random_forest", "rbf", "rsm", "xgboost"]
ALGORITMOS = ["age", "de", "shade"]
ALGO_LABELS = {"age": "AGE", "de": "DE", "shade": "SHADE"}

# Pares de bloques con la misma ventana de validación
# clave = hito de entrenamiento (eje x), valor = (bloque_no_acum, bloque_acum)
BLOQUES_PARES = {
    20: ("1-20",  "1-20"),
    40: ("21-40", "1-40"),
    60: ("41-60", "1-60"),
    80: ("61-80", "1-80"),
}

COLORS = {
    "no_acumulativo": "#1f77b4",
    "acumulativo":    "#ff7f0e",
}
LABELS = {
    "no_acumulativo": "No acumulativa",
    "acumulativo":    "Acumulativa",
}


def cargar_datos() -> pd.DataFrame:
    rows = []
    for modelo in MODELOS:
        for estrategia in ("no_acumulativo", "acumulativo"):
            base_est = BASE_DIR / modelo / estrategia
            if not base_est.exists():
                continue
            for funcion_dir in sorted(base_est.iterdir()):
                funcion = funcion_dir.name
                for algoritmo in ALGORITMOS:
                    alg_dir = funcion_dir / algoritmo / modelo
                    if not alg_dir.exists():
                        continue
                    for hito, (bloque_no, bloque_ac) in BLOQUES_PARES.items():
                        bloque = bloque_no if estrategia == "no_acumulativo" else bloque_ac
                        json_path = alg_dir / bloque / f"{modelo}_metricas.json"
                        if not json_path.exists():
                            continue
                        data = json.loads(json_path.read_text(encoding="utf-8"))
                        rows.append({
                            "modelo":     modelo,
                            "estrategia": estrategia,
                            "funcion":    funcion,
                            "algoritmo":  algoritmo,
                            "hito":       hito,
                            "spearman":   float(data.get("spearman") or 0),
                        })
    return pd.DataFrame(rows)


def main() -> None:
    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "font.size":        10.5,
        "axes.labelsize":   11,
        "axes.titlesize":   12,
        "legend.fontsize":  9.5,
        "axes.edgecolor":   "#444444",
        "axes.labelcolor":  "#222222",
        "xtick.color":      "#444444",
        "ytick.color":      "#444444",
    })

    print("Cargando datos...")
    df = cargar_datos()
    print(f"  {len(df)} filas cargadas.")

    # Media de Spearman por (algoritmo, estrategia, hito) sobre modelos × funciones
    agg = (
        df.groupby(["algoritmo", "estrategia", "hito"])["spearman"]
        .mean()
        .reset_index()
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    x = [20, 40, 60, 80]

    for ax, algoritmo in zip(axes, ALGORITMOS):
        df_algo = agg[agg["algoritmo"] == algoritmo]
        for estrategia in ("no_acumulativo", "acumulativo"):
            df_est = df_algo[df_algo["estrategia"] == estrategia].sort_values("hito")
            y = df_est["spearman"].tolist()
            ax.plot(
                x, y,
                marker="o",
                linewidth=2.0,
                markersize=5.5,
                color=COLORS[estrategia],
                label=LABELS[estrategia],
            )

        ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_title(ALGO_LABELS[algoritmo])
        ax.set_xlabel("Presupuesto usado para entrenamiento (%)")
        ax.set_xticks(x)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Spearman medio")

    # Leyenda compartida debajo de los 3 paneles
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#dddddd",
        bbox_to_anchor=(0.5, -0.08),
    )

    fig.tight_layout(rect=(0, 0.06, 1, 1))

    png = OUT_DIR / "evolucion_spearman_por_bloque_por_algoritmo.png"
    pdf = OUT_DIR / "evolucion_spearman_por_bloque_por_algoritmo.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Generado: {png}")
    print(f"Generado: {pdf}")

    # CSV de respaldo
    csv_path = OUT_DIR / "evolucion_spearman_por_bloque_por_algoritmo.csv"
    agg.to_csv(csv_path, index=False)
    print(f"Datos: {csv_path}")

    # Figuras individuales por algoritmo (para subfigura en LaTeX)
    y_all = agg["spearman"].tolist()
    y_margin = (max(y_all) - min(y_all)) * 0.08
    y_lim = (min(y_all) - y_margin, max(y_all) + y_margin)

    for algoritmo in ALGORITMOS:
        fig_i, ax_i = plt.subplots(figsize=(9.0, 2.8))
        df_algo = agg[agg["algoritmo"] == algoritmo]
        for estrategia in ("no_acumulativo", "acumulativo"):
            df_est = df_algo[df_algo["estrategia"] == estrategia].sort_values("hito")
            y = df_est["spearman"].tolist()
            ax_i.plot(
                x, y,
                marker="o",
                linewidth=2.0,
                markersize=5.5,
                color=COLORS[estrategia],
                label=LABELS[estrategia],
            )
        ax_i.axhline(0, color="#888888", linewidth=0.8, linestyle="--", alpha=0.6)
        ax_i.set_xlabel("Presupuesto usado para entrenamiento (%)")
        ax_i.set_ylabel("Spearman medio")
        ax_i.set_xticks(x)
        ax_i.set_ylim(y_lim)
        ax_i.grid(axis="y", alpha=0.25, linestyle="--")
        ax_i.spines["top"].set_visible(False)
        ax_i.spines["right"].set_visible(False)
        ax_i.legend(frameon=True, facecolor="white", edgecolor="#dddddd", fontsize=9)
        fig_i.tight_layout()
        for ext in ("png", "pdf"):
            out_i = OUT_DIR / f"evolucion_spearman_por_bloque_{algoritmo}.{ext}"
            fig_i.savefig(out_i, dpi=300, bbox_inches="tight")
            print(f"Generado: {out_i}")
        plt.close(fig_i)


if __name__ == "__main__":
    main()
