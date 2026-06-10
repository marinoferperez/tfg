from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_BASE_DIR = (
    "results/cec/cec2017_d10_tam50_reinicio_seleccionado/"
    "benchmarking_surrogates_offline_next/future_next"
)
DEFAULT_OUT_DIR = "memoria/figuras/surrogates_offline"

MODEL_LABELS = {
    "hgb": "HGB",
    "lasso": "LASSO",
    "mlp": "MLP",
    "random_forest": "Random Forest",
    "rbf": "RBF",
    "rsm": "RSM",
    "svr": "SVR",
    "xgboost": "XGBoost",
}

MODEL_COLORS = {
    "lasso": "#4E79A7",
    "rsm": "#F28E2B",
    "mlp": "#E15759",
    "random_forest": "#72B7B2",
    "rbf": "#54A24B",
    "hgb": "#B07AA1",
    "xgboost": "#FF9DA6",
    "svr": "#111111",
}

BLOCKS = {
    "acumulativo": ["1-20", "1-40", "1-60", "1-80"],
    "no_acumulativo": ["1-20", "21-40", "41-60", "61-80"],
}

ORDER_MODELS = [
    "lasso",
    "rsm",
    "mlp",
    "random_forest",
    "rbf",
    "hgb",
    "xgboost",
    "svr",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera graficos de evolucion temporal del rendimiento medio de los "
            "modelos surrogate para las estrategias acumulativa y no acumulativa."
        )
    )
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--metric",
        default="spearman",
        choices=["spearman", "nmae", "nrmse", "train_time_s"],
    )
    parser.add_argument(
        "--exclude-models",
        nargs="*",
        default=[],
        help="Modelos a excluir en ambas estrategias.",
    )
    parser.add_argument(
        "--exclude-models-acumulativo",
        nargs="*",
        default=["svr"],
        help="Modelos a excluir solo en la estrategia acumulativa.",
    )
    parser.add_argument(
        "--exclude-models-no-acumulativo",
        nargs="*",
        default=[],
        help="Modelos a excluir solo en la estrategia no acumulativa.",
    )
    parser.add_argument(
        "--prefix",
        default="evolucion_spearman_modelos",
        help="Prefijo de los ficheros de salida.",
    )
    return parser.parse_args()


def load_metrics(base_dir: Path, metric: str) -> pd.DataFrame:
    rows = []
    for path in sorted(base_dir.rglob("*_metricas.json")):
        rel = path.relative_to(base_dir).parts
        if len(rel) < 7:
            continue
        model, protocol, function, algorithm = rel[:4]
        block = rel[-2]
        if block == model:
            continue
        if protocol not in BLOCKS or block not in BLOCKS[protocol]:
            continue

        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "modelo": model,
                "estrategia": protocol,
                "funcion": function,
                "algoritmo": algorithm,
                "bloque": block,
                "valor": data.get(metric),
                "n_seeds_evaluadas": data.get("n_seeds_evaluadas"),
            }
        )
    if not rows:
        raise SystemExit(f"No se encontraron metricas temporales en {base_dir}")
    return pd.DataFrame(rows)


def filter_strategy_models(
    df: pd.DataFrame,
    strategy: str,
    exclude_common: set[str],
    exclude_strategy: set[str],
) -> pd.DataFrame:
    strategy_df = df[df["estrategia"] == strategy].copy()
    excluded = exclude_common | exclude_strategy
    if excluded:
        strategy_df = strategy_df[~strategy_df["modelo"].isin(excluded)]
    return strategy_df


def aggregate(df: pd.DataFrame, by_algorithm: bool = False) -> pd.DataFrame:
    group_cols = ["estrategia", "modelo", "bloque"]
    if by_algorithm:
        group_cols = ["estrategia", "modelo", "bloque", "algoritmo"]
    return (
        df.groupby(group_cols, as_index=False)
        .agg(valor=("valor", "mean"), n_metricas=("valor", "count"))
        .sort_values(group_cols)
    )


def plot_strategy_by_algorithm(
    agg_df: pd.DataFrame,
    strategy: str,
    metric: str,
    out_dir: Path,
    prefix: str,
) -> list[Path]:
    """Genera una figura independiente por algoritmo (para uso como subfiguras)."""
    algoritmos = sorted(agg_df["algoritmo"].unique())
    outputs = []
    for algoritmo in algoritmos:
        df_algo = agg_df[
            (agg_df["estrategia"] == strategy) & (agg_df["algoritmo"] == algoritmo)
        ]
        out_path = _plot_single(
            df_algo, strategy, metric, out_dir,
            prefix=f"{prefix}_{strategy}_{algoritmo}",
            title=algoritmo.upper(),
        )
        outputs.append(out_path)
    return outputs


def _plot_single(
    agg_df: pd.DataFrame,
    strategy: str,
    metric: str,
    out_dir: Path,
    prefix: str,
    title: str | None = None,
) -> Path:
    metric_label = {
        "spearman": "Spearman medio",
        "nmae": "nMAE medio",
        "nrmse": "nRMSE medio",
        "train_time_s": "Tiempo medio de entrenamiento (s)",
    }[metric]

    blocks = BLOCKS[strategy]
    x = [20, 40, 60, 80]

    fig, ax = plt.subplots(figsize=(7.4, 5.1))
    models = [m for m in ORDER_MODELS if m in set(agg_df["modelo"])]
    for model in models:
        model_df = agg_df[agg_df["modelo"] == model].set_index("bloque")
        y = [model_df.loc[block, "valor"] if block in model_df.index else None for block in blocks]
        ax.plot(
            x, y,
            marker="o",
            linewidth=1.9,
            markersize=5.2,
            color=MODEL_COLORS.get(model, None),
            label=MODEL_LABELS.get(model, model),
        )

    ax.axhline(0, color="#777777", linewidth=0.8, linestyle="--", alpha=0.65)
    if title:
        ax.set_title(title)
    ax.set_xlabel("Presupuesto usado para entrenamiento (%)")
    ax.set_ylabel(metric_label)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if metric == "spearman":
        values = agg_df["valor"].dropna()
        ymin = min(-0.2, values.min() - 0.05)
        ymax = min(1.0, max(0.65, values.max() + 0.05))
        ax.set_ylim(ymin, ymax)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=4,
        frameon=True,
        facecolor="white",
        edgecolor="#dddddd",
    )
    fig.tight_layout(rect=(0, 0.20, 1, 1))

    out_path = out_dir / f"{prefix}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_strategy(
    agg_df: pd.DataFrame,
    strategy: str,
    metric: str,
    out_dir: Path,
    prefix: str,
) -> Path:
    strategy_name = {
        "acumulativo": "Estrategia acumulativa",
        "no_acumulativo": "Estrategia no acumulativa",
    }[strategy]
    metric_label = {
        "spearman": "Spearman medio",
        "nmae": "nMAE medio",
        "nrmse": "nRMSE medio",
        "train_time_s": "Tiempo medio de entrenamiento (s)",
    }[metric]

    blocks = BLOCKS[strategy]
    x = [20, 40, 60, 80]

    fig, ax = plt.subplots(figsize=(7.4, 5.1))
    models = [m for m in ORDER_MODELS if m in set(agg_df["modelo"])]
    for model in models:
        model_df = agg_df[agg_df["modelo"] == model].set_index("bloque")
        y = [model_df.loc[block, "valor"] if block in model_df.index else None for block in blocks]
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.9,
            markersize=5.2,
            color=MODEL_COLORS.get(model, None),
            label=MODEL_LABELS.get(model, model),
        )

    ax.axhline(0, color="#777777", linewidth=0.8, linestyle="--", alpha=0.65)
    ax.set_title(strategy_name)
    ax.set_xlabel("Presupuesto usado para entrenamiento (%)")
    ax.set_ylabel(metric_label)
    ax.set_xticks(x)
    ax.set_xticklabels([str(value) for value in x])
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if metric == "spearman":
        values = agg_df["valor"].dropna()
        ymin = min(-0.2, values.min() - 0.05)
        ymax = min(1.0, max(0.65, values.max() + 0.05))
        ax.set_ylim(ymin, ymax)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.27),
        ncol=4,
        frameon=True,
        facecolor="white",
        edgecolor="#dddddd",
    )
    fig.tight_layout(rect=(0, 0.18, 1, 1))

    out_path = out_dir / f"{prefix}_{strategy}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 8.6,
            "axes.edgecolor": "#444444",
            "axes.labelcolor": "#222222",
            "xtick.color": "#444444",
            "ytick.color": "#444444",
        }
    )

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_metrics(base_dir, args.metric)
    exclude_common = set(args.exclude_models)
    strategy_excludes = {
        "acumulativo": set(args.exclude_models_acumulativo),
        "no_acumulativo": set(args.exclude_models_no_acumulativo),
    }

    outputs = []
    summary_frames = []
    for strategy in ("acumulativo", "no_acumulativo"):
        strategy_df = filter_strategy_models(
            df,
            strategy,
            exclude_common=exclude_common,
            exclude_strategy=strategy_excludes[strategy],
        )
        # Figura original agregada (sin cambios)
        agg_df = aggregate(strategy_df)
        summary_frames.append(agg_df)
        outputs.append(plot_strategy(agg_df, strategy, args.metric, out_dir, args.prefix))

        # Figuras por algoritmo (subfiguras independientes)
        agg_df_algo = aggregate(strategy_df, by_algorithm=True)
        algo_outputs = plot_strategy_by_algorithm(
            agg_df_algo, strategy, args.metric, out_dir, args.prefix
        )
        outputs.extend(algo_outputs)

    summary = pd.concat(summary_frames, ignore_index=True)
    summary_path = out_dir / f"{args.prefix}_datos.csv"
    summary.to_csv(summary_path, index=False)

    print("Graficos generados:")
    for output in outputs:
        print(f"  {output}")
    print(f"Datos agregados: {summary_path}")


if __name__ == "__main__":
    main()
