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
OUT_DIR = ROOT / "memoria" / "figuras" / "surrogates_online"

MAX_EVALS = 100_000
BUDGET_GRID = np.linspace(0.0, 100.0, 201)
EVAL_GRID = BUDGET_GRID * MAX_EVALS / 100.0
EPS = 1e-12

CONFIGS = {
    "age": {
        "base_dir": "online_final_base_age",
        "rbf_dir": "online_final_surrogate_age",
        "base_label": "AGE-1",
        "rbf_label": "AGE-1-RBF",
        "rbf_color": "#4C78A8",
    },
    "de": {
        "base_dir": "online_final_base_de",
        "rbf_dir": "online_final_surrogate_de",
        "base_label": "DE-10",
        "rbf_label": "DE-10-RBF",
        "rbf_color": "#F58518",
    },
    "shade": {
        "base_dir": "online_final_base_shade",
        "rbf_dir": "online_final_surrogate_shade",
        "base_label": "SHADE-10",
        "rbf_label": "SHADE-10-RBF",
        "rbf_color": "#54A24B",
    },
}


def metric_files(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "metricas_runs").rglob("resultados_*.csv"))


def curve_from_csv(csv_path: Path, func_id: int) -> np.ndarray | None:
    optimum = 100.0 * func_id
    values: list[float] = []
    current = None
    j = 0

    try:
        chunks = pd.read_csv(
            csv_path,
            usecols=["evaluaciones", "min/mejor_hasta_ahora"],
            chunksize=200_000,
        )
    except ValueError:
        return None

    for chunk in chunks:
        if chunk.empty:
            continue

        evaluaciones = chunk["evaluaciones"].to_numpy(dtype=float)
        errores = np.maximum(
            chunk["min/mejor_hasta_ahora"].to_numpy(dtype=float) - optimum,
            EPS,
        )
        current = float(errores[-1])

        while j < len(EVAL_GRID) and EVAL_GRID[j] <= evaluaciones[-1]:
            idx = int(np.searchsorted(evaluaciones, EVAL_GRID[j], side="left"))
            if idx >= len(errores):
                idx = len(errores) - 1
            values.append(float(errores[idx]))
            j += 1

        if j >= len(EVAL_GRID):
            break

    if current is None:
        return None

    while j < len(EVAL_GRID):
        values.append(current)
        j += 1

    curve = np.asarray(values, dtype=float)
    return curve / max(curve[0], EPS)


def mean_function_curve(run_dir: Path, func_id: int) -> np.ndarray | None:
    curves = []
    for csv_path in metric_files(run_dir / f"f{func_id}"):
        curve = curve_from_csv(csv_path, func_id)
        if curve is not None:
            curves.append(curve)

    if not curves:
        return None

    return np.mean(np.vstack(curves), axis=0)


def aggregate_curve(run_dir: Path) -> np.ndarray:
    curves = []
    for func_id in range(1, 31):
        curve = mean_function_curve(run_dir, func_id)
        if curve is not None:
            curves.append(curve)

    if not curves:
        raise FileNotFoundError(f"No se encontraron curvas en {run_dir}")

    return np.mean(np.vstack(curves), axis=0)


def plot_algo(algo: str, cfg: dict[str, str]) -> None:
    base_curve = aggregate_curve(RESULTS_DIR / cfg["base_dir"])
    rbf_curve = aggregate_curve(RESULTS_DIR / cfg["rbf_dir"])

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(
        BUDGET_GRID,
        base_curve,
        color="#7F7F7F",
        linestyle=":",
        linewidth=2.2,
        label=cfg["base_label"],
    )
    ax.plot(
        BUDGET_GRID,
        rbf_curve,
        color=cfg["rbf_color"],
        linewidth=2.2,
        label=cfg["rbf_label"],
    )

    ax.set_yscale("log")
    ax.set_xlabel("Presupuesto (%)")
    ax.set_ylabel("Error relativo medio")
    ax.set_xlim(0, 100)
    ax.grid(False, which="both", axis="both")
    ax.legend(frameon=True, fontsize=9, loc="best")
    fig.tight_layout(pad=0.6)

    for ext in ("png", "pdf"):
        fig.savefig(
            OUT_DIR / f"convergencia_online_agregada_{algo}.{ext}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)

    pd.DataFrame(
        {
            "presupuesto": BUDGET_GRID,
            cfg["base_label"]: base_curve,
            cfg["rbf_label"]: rbf_curve,
        }
    ).to_csv(OUT_DIR / f"convergencia_online_agregada_{algo}.csv", index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for algo, cfg in CONFIGS.items():
        plot_algo(algo, cfg)


if __name__ == "__main__":
    main()
