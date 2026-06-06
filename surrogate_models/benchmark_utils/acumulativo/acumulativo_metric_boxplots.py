from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
cache_dir = ROOT / "tmp" / ".mplconfig"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_BENCHMARK_ROOT = (
    ROOT
    / "results"
    / "cec"
    / "experimentos_mhs_ambos_cec2017_d10_tam50"
    / "benchmarking"
    / "acumulativo"
)

METRICS = (
    ("mae", "MAE"),
    ("nmae", "nMAE"),
    ("rmse", "RMSE"),
    ("nrmse", "nRMSE"),
    ("spearman", "Spearman"),
)

FUNCTIONS = ("f1", "f3", "f10", "f17", "f29")
ALGORITHMS = (("age", "AGE"), ("de", "DE"))
MODELS = (
    ("lasso", "LASSO"),
    ("mlp", "MLP"),
    ("svr", "SVR"),
    ("random_forest", "Random Forest"),
    ("rbf", "RBF"),
    ("rsm", "RSM"),
    ("hgb", "HGB"),
    ("xgboost", "XGBoost"),
)
MODEL_COLORS = {
    "lasso": "#4C78A8",
    "mlp": "#F58518",
    "svr": "#E45756",
    "random_forest": "#72B7B2",
    "rbf": "#54A24B",
    "rsm": "#EECA3B",
    "hgb": "#B279A2",
    "xgboost": "#FF9DA6",
}
BATCHES = (
    ("1-20", "1-20%"),
    ("1-40", "1-40%"),
    ("1-60", "1-60%"),
    ("1-80", "1-80%"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera boxplots por seed para MAE, RMSE, R² y Spearman "
            "bajo la estrategia acumulativa."
        )
    )
    parser.add_argument(
        "--benchmark-root",
        default=str(DEFAULT_BENCHMARK_ROOT),
        help="Directorio raiz del benchmark acumulativo.",
    )
    parser.add_argument(
        "--functions",
        nargs="*",
        default=list(FUNCTIONS),
        help="Funciones a procesar.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="*",
        default=[key for key, _ in ALGORITHMS],
        help="Algoritmos a procesar.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=[key for key, _ in MODELS],
        help="Modelos a representar en las figuras.",
    )
    parser.add_argument(
        "--batches",
        nargs="*",
        default=[key for key, _ in BATCHES],
        help="Batches a representar.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=["rmse", "spearman"],
        help="Metricas a representar. Si se indican varias, se dibujan alineadas en una sola figura por batch.",
    )
    return parser.parse_args()


def ensure_matplotlib_cache() -> None:
    cache_dir = ROOT / "tmp" / ".mplconfig"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))


def read_metric_series(model_dir: Path, batch_dir: str, metric: str) -> list[float]:
    csv_path = model_dir / batch_dir / f"{model_dir.name}_runs.csv"
    if not csv_path.is_file():
        return []
    df = pd.read_csv(csv_path)
    return pd.to_numeric(df[metric], errors="coerce").dropna().tolist()


def render_metric_figure(
    *,
    function_id: str,
    algorithm_label: str,
    algorithm_dir: Path,
    batches: list[tuple[str, str]],
    metrics: list[tuple[str, str]],
    selected_models: list[tuple[str, str]],
) -> list[Path]:
    out_paths: list[Path] = []
    model_positions = np.arange(len(selected_models)) + 1

    for batch_dir, batch_label in batches:
        fig, axes = plt.subplots(
            1,
            len(metrics),
            figsize=(6.0 * len(metrics), 5.8),
            squeeze=False,
        )
        axes_row = axes[0]
        for ax, (metric, metric_label) in zip(axes_row, metrics):
            data = [
                read_metric_series(algorithm_dir / model_dir, batch_dir, metric)
                for model_dir, _ in selected_models
            ]
            bp = ax.boxplot(
                data,
                positions=model_positions,
                widths=0.58,
                patch_artist=True,
                medianprops=dict(color="black", linewidth=1.0),
                boxprops=dict(linewidth=0.9),
                whiskerprops=dict(linewidth=0.9),
                capprops=dict(linewidth=0.9),
                flierprops=dict(
                    marker="o",
                    markersize=2.6,
                    markerfacecolor="black",
                    markeredgecolor="black",
                    alpha=0.45,
                ),
            )
            for patch, (model_key, _) in zip(bp["boxes"], selected_models):
                patch.set_facecolor(MODEL_COLORS.get(model_key, "#4063D8"))
                patch.set_alpha(0.72)

            ax.set_title(metric_label, fontsize=12)
            ax.set_ylabel(metric_label)
            ax.set_xticks(model_positions)
            ax.set_xticklabels([label for _, label in selected_models], rotation=22, ha="right")
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            if metric in {"spearman"}:
                ax.axhline(0.0, color="gray", linewidth=0.9, linestyle=":")

        fig.suptitle(
            f"{function_id} con {algorithm_label}: distribucion por seed para el batch {batch_label}",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))

        metrics_slug = "_".join(metric for metric, _ in metrics)
        out_path = algorithm_dir / f"boxplot_batch_{batch_dir}_{metrics_slug}_por_surrogate.png"
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(out_path)

    return out_paths


def main() -> None:
    ensure_matplotlib_cache()
    args = parse_args()
    benchmark_root = Path(args.benchmark_root).resolve()
    algorithm_map = dict(ALGORITHMS)
    selected_models = [(key, label) for key, label in MODELS if key in set(args.models)]
    selected_metrics = [(key, label) for key, label in METRICS if key in set(args.metrics)]
    selected_batches = [(key, label) for key, label in BATCHES if key in set(args.batches)]
    if not selected_models:
        raise SystemExit("No se ha seleccionado ningun modelo valido en --models.")
    if not selected_metrics:
        raise SystemExit("No se ha seleccionado ninguna metrica valida en --metrics.")
    if not selected_batches:
        raise SystemExit("No se ha seleccionado ningun batch valido en --batches.")

    for function_id in args.functions:
        function_dir = benchmark_root / function_id
        if not function_dir.is_dir():
            continue
        for algorithm_key in args.algorithms:
            algorithm_dir = function_dir / algorithm_key
            if not algorithm_dir.is_dir():
                continue
            algorithm_label = algorithm_map.get(algorithm_key, algorithm_key.upper())
            out_paths = render_metric_figure(
                function_id=function_id,
                algorithm_label=algorithm_label,
                algorithm_dir=algorithm_dir,
                batches=selected_batches,
                metrics=selected_metrics,
                selected_models=selected_models,
            )
            for out_path in out_paths:
                print(out_path)


if __name__ == "__main__":
    main()
