from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("default")

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "results" / "cec"
OUT_DIR = ROOT / "memoria" / "figuras" / "surrogates_online" / "convergencia_subfiguras"

MAX_EVALS = 100_000
EPS = 1e-12
BUDGET_GRID = np.linspace(0.0, 100.0, 201)

CONFIGS = {
    "age": {
        "base_dir": "online_final_base_age",
        "rbf_dir": "online_final_surrogate_age",
        "base_label": "AGE-1",
        "rbf_label": "AGE-1-RBF",
        "rbf_color": "#4C78A8",
        "funcs": [3, 9, 22],
    },
    "de": {
        "base_dir": "online_final_base_de",
        "rbf_dir": "online_final_surrogate_de",
        "base_label": "DE-10",
        "rbf_label": "DE-10-RBF",
        "rbf_color": "#F58518",
        "funcs": [4, 5, 10],
    },
    "shade": {
        "base_dir": "online_final_base_shade",
        "rbf_dir": "online_final_surrogate_shade",
        "base_label": "SHADE-10",
        "rbf_label": "SHADE-10-RBF",
        "rbf_color": "#54A24B",
        "funcs": [1, 14, 30],
    },
}


def metric_files(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "metricas_runs").rglob("resultados_*.csv"))


def mean_curve(run_dir: Path, func_id: int) -> pd.DataFrame:
    fdir = run_dir / f"f{func_id}"
    aligned_curves = []
    optimum = 100.0 * func_id

    for csv_path in metric_files(fdir):
        df = pd.read_csv(csv_path)
        if "evaluaciones" not in df or "min/mejor_hasta_ahora" not in df:
            continue
        curve = df[["evaluaciones", "min/mejor_hasta_ahora"]].copy()
        curve["presupuesto"] = 100.0 * curve["evaluaciones"] / MAX_EVALS
        curve["error"] = np.maximum(curve["min/mejor_hasta_ahora"] - optimum, EPS)
        curve = curve.sort_values("presupuesto")

        x = curve["presupuesto"].to_numpy()
        y = curve["error"].to_numpy()
        idx = np.searchsorted(x, BUDGET_GRID, side="right") - 1
        idx = np.clip(idx, 0, len(y) - 1)
        aligned_curves.append(y[idx])

    if not aligned_curves:
        raise FileNotFoundError(f"No se encontraron curvas para {fdir}")

    mean_error = np.mean(np.vstack(aligned_curves), axis=0)
    return pd.DataFrame({"presupuesto": BUDGET_GRID, "error": mean_error})


def plot_func(algo: str, func_id: int, cfg: dict[str, object]) -> None:
    base_curve = mean_curve(RESULTS_DIR / str(cfg["base_dir"]), func_id)
    rbf_curve = mean_curve(RESULTS_DIR / str(cfg["rbf_dir"]), func_id)

    fig, ax = plt.subplots(figsize=(9.0, 2.8))
    ax.plot(
        base_curve["presupuesto"],
        base_curve["error"],
        color="#7F7F7F",
        linestyle=":",
        linewidth=2.0,
        label=str(cfg["base_label"]),
    )
    ax.plot(
        rbf_curve["presupuesto"],
        rbf_curve["error"],
        color=str(cfg["rbf_color"]),
        linewidth=2.0,
        label=str(cfg["rbf_label"]),
    )

    ax.set_yscale("log")
    ax.set_xlabel("Presupuesto (%)")
    ax.set_ylabel("")
    ax.grid(False, which="both", axis="both")
    ax.legend(frameon=True, fontsize=8, loc="best")
    ax.set_xlim(0, 100)

    fig.tight_layout(pad=0.6)
    for ext in ("png", "pdf"):
        fig.savefig(
            OUT_DIR / f"convergencia_online_{algo}_f{func_id}.{ext}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for algo, cfg in CONFIGS.items():
        for func_id in cfg["funcs"]:
            plot_func(algo, func_id, cfg)


if __name__ == "__main__":
    main()
