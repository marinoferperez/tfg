from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(
    "results/cec/cec2017_d10_tam50_reinicio_seleccionado/"
    "benchmarking_surrogates_offline_next/future_next"
)
OUT_DIR = Path("memoria/figuras/surrogates_offline")
TABLE_DIR = Path("memoria/tablas")
FUNCIONES_ESPERADAS = ("f1", "f4", "f10", "f12", "f18", "f22", "f29")
ALGORITMOS = ("age", "de", "shade")
BLOQUES = {
    "no_acumulativo": ("1-20", "21-40", "41-60", "61-80"),
    "acumulativo": ("1-20", "1-40", "1-60", "1-80"),
}
X_PCT = {
    "1-20": 20,
    "21-40": 40,
    "41-60": 60,
    "61-80": 80,
    "1-40": 40,
    "1-60": 60,
    "1-80": 80,
}


def load_svr_metrics() -> pd.DataFrame:
    rows = []
    for path in sorted((BASE_DIR / "svr").rglob("svr_metricas.json")):
        rel = path.relative_to(BASE_DIR).parts
        if len(rel) < 7:
            continue
        model, protocol, function, algorithm = rel[:4]
        block = rel[-2]
        if model != "svr" or block == model:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "estrategia": protocol,
                "funcion": function,
                "algoritmo": algorithm,
                "bloque": block,
                "x_pct": X_PCT[block],
                "train_time_s": data["train_time_s"],
                "train_time_s_std": data.get("train_time_s_std"),
                "spearman": data.get("spearman"),
                "n_train": data.get("n_train"),
                "n_seeds_evaluadas": data.get("n_seeds_evaluadas"),
            }
        )
    if not rows:
        raise SystemExit("No se encontraron metricas de SVR.")
    return pd.DataFrame(rows)


def expected_count(strategy: str) -> int:
    return len(FUNCIONES_ESPERADAS) * len(ALGORITMOS) * len(BLOQUES[strategy])


def write_coverage_table(df: pd.DataFrame) -> Path:
    common = df[df["funcion"].isin(["f1", "f4"])].copy()
    time_summary = (
        common.groupby("estrategia", as_index=False)
        .agg(
            train_time_s_mean=("train_time_s", "mean"),
            train_time_s_max=("train_time_s", "max"),
        )
    )
    coverage = (
        df.groupby("estrategia", as_index=False)
        .agg(n_metricas=("train_time_s", "count"))
    )
    coverage["n_metricas_esperadas"] = coverage["estrategia"].map(expected_count)
    coverage["cobertura_pct"] = (
        100.0 * coverage["n_metricas"] / coverage["n_metricas_esperadas"]
    )
    summary = coverage.merge(time_summary, on="estrategia", how="left")
    summary = summary.sort_values("estrategia")

    csv_path = OUT_DIR / "viabilidad_svr_resumen.csv"
    summary.to_csv(csv_path, index=False)

    labels = {
        "acumulativo": "Acumulativa",
        "no_acumulativo": "No acumulativa",
    }
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        (
            r"\textbf{Estrategia} & \textbf{Ejecuciones} & "
            r"\textbf{Cobertura} & \textbf{Entren. medio (s)} & "
            r"\textbf{Entren. máx. (s)} \\"
        ),
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            " & ".join(
                [
                    labels[row["estrategia"]],
                    f"{int(row['n_metricas'])}/{int(row['n_metricas_esperadas'])}",
                    f"{row['cobertura_pct']:.1f}\\%",
                    f"{row['train_time_s_mean']:.3f}",
                    f"{row['train_time_s_max']:.3f}",
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    tex_path = TABLE_DIR / "viabilidad_svr.tex"
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    return tex_path


def plot_training_time(df: pd.DataFrame) -> Path:
    common = df[df["funcion"].isin(["f1", "f4"])].copy()
    agg = (
        common.groupby(["estrategia", "x_pct"], as_index=False)
        .agg(train_time_s=("train_time_s", "mean"))
        .sort_values(["estrategia", "x_pct"])
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "axes.edgecolor": "#444444",
            "axes.labelcolor": "#222222",
            "xtick.color": "#444444",
            "ytick.color": "#444444",
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    styles = {
        "no_acumulativo": {
            "label": "No acumulativa",
            "color": "#4E79A7",
            "marker": "o",
            "linestyle": "-",
        },
        "acumulativo": {
            "label": "Acumulativa",
            "color": "#F28E2B",
            "marker": "s",
            "linestyle": "--",
        },
    }

    for strategy, strategy_df in agg.groupby("estrategia", sort=False):
        style = styles[strategy]
        ax.plot(
            strategy_df["x_pct"],
            strategy_df["train_time_s"],
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.2,
            markersize=6.5,
        )

    ax.set_xlabel("Presupuesto usado para entrenamiento (%)")
    ax.set_ylabel("Tiempo medio de entrenamiento (s)")
    ax.set_xticks([20, 40, 60, 80])
    ax.grid(axis="y", alpha=0.35, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#dddddd")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "viabilidad_svr_tiempo_entrenamiento.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    agg.to_csv(OUT_DIR / "viabilidad_svr_tiempo_entrenamiento.csv", index=False)
    return out_path


def plot_training_time_for_function(df: pd.DataFrame, function: str) -> Path:
    function_df = df[df["funcion"] == function].copy()
    if function_df.empty:
        raise SystemExit(f"No hay metricas de SVR para {function}.")

    agg = (
        function_df.groupby(["estrategia", "x_pct"], as_index=False)
        .agg(
            train_time_s=("train_time_s", "mean"),
            train_time_s_max=("train_time_s", "max"),
        )
        .sort_values(["estrategia", "x_pct"])
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "axes.edgecolor": "#444444",
            "axes.labelcolor": "#222222",
            "xtick.color": "#444444",
            "ytick.color": "#444444",
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    styles = {
        "no_acumulativo": {
            "label": "No acumulativa",
            "color": "#4E79A7",
            "marker": "o",
            "linestyle": "-",
        },
        "acumulativo": {
            "label": "Acumulativa",
            "color": "#F28E2B",
            "marker": "s",
            "linestyle": "--",
        },
    }

    for strategy, strategy_df in agg.groupby("estrategia", sort=False):
        style = styles[strategy]
        ax.plot(
            strategy_df["x_pct"],
            strategy_df["train_time_s"],
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.2,
            markersize=6.5,
        )

    ax.set_xlabel("Presupuesto usado para entrenamiento (%)")
    ax.set_ylabel("Tiempo medio de entrenamiento (s)")
    ax.set_xticks([20, 40, 60, 80])
    ax.grid(axis="y", alpha=0.35, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#dddddd")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"viabilidad_svr_tiempo_entrenamiento_{function}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    agg.to_csv(OUT_DIR / f"viabilidad_svr_tiempo_entrenamiento_{function}.csv", index=False)
    return out_path


def main() -> None:
    df = load_svr_metrics()
    table_path = write_coverage_table(df)
    plot_path = plot_training_time(df)
    plot_f1_path = plot_training_time_for_function(df, "f1")
    plot_f4_path = plot_training_time_for_function(df, "f4")
    print(f"Tabla: {table_path}")
    print(f"Figura: {plot_path}")
    print(f"Figura PDF: {plot_path.with_suffix('.pdf')}")
    print(f"Figura f1: {plot_f1_path}")
    print(f"Figura f1 PDF: {plot_f1_path.with_suffix('.pdf')}")
    print(f"Figura f4: {plot_f4_path}")
    print(f"Figura f4 PDF: {plot_f4_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
