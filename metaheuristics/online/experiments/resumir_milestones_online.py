from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "results" / "cec"
OUT_CSV = ROOT / "memoria" / "figuras" / "surrogates_online" / "milestones_online_por_funcion.csv"

MAX_EVALS = 100_000
MILESTONES = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
EVAL_MILESTONES = MILESTONES * MAX_EVALS / 100.0
EPS = 1e-12

CONFIGS = {
    "AGE": {
        "base_dir": "online_final_base_age",
        "rbf_dir": "online_final_surrogate_age",
        "base_label": "AGE-1",
        "rbf_label": "AGE-1-RBF",
    },
    "DE": {
        "base_dir": "online_final_base_de",
        "rbf_dir": "online_final_surrogate_de",
        "base_label": "DE-10",
        "rbf_label": "DE-10-RBF",
    },
    "SHADE": {
        "base_dir": "online_final_base_shade",
        "rbf_dir": "online_final_surrogate_shade",
        "base_label": "SHADE-10",
        "rbf_label": "SHADE-10-RBF",
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

        while j < len(EVAL_MILESTONES) and EVAL_MILESTONES[j] <= evaluaciones[-1]:
            idx = int(np.searchsorted(evaluaciones, EVAL_MILESTONES[j], side="left"))
            if idx >= len(errores):
                idx = len(errores) - 1
            values.append(float(errores[idx]))
            j += 1

        if j >= len(EVAL_MILESTONES):
            break

    if current is None:
        return None

    while j < len(EVAL_MILESTONES):
        values.append(current)
        j += 1

    return np.asarray(values, dtype=float)


def mean_curve(run_dir: Path, func_id: int) -> np.ndarray | None:
    curves = []
    for csv_path in metric_files(run_dir / f"f{func_id}"):
        curve = curve_from_csv(csv_path, func_id)
        if curve is not None:
            curves.append(curve)
    if not curves:
        return None
    return np.mean(np.vstack(curves), axis=0)


def main() -> None:
    rows = []
    for algo, cfg in CONFIGS.items():
        base_root = RESULTS_DIR / cfg["base_dir"]
        rbf_root = RESULTS_DIR / cfg["rbf_dir"]
        for func_id in range(1, 31):
            base = mean_curve(base_root, func_id)
            rbf = mean_curve(rbf_root, func_id)
            if base is None or rbf is None:
                continue
            ratio = rbf / np.maximum(base, EPS)
            rows.append(
                {
                    "algoritmo": algo,
                    "funcion": func_id,
                    "base_label": cfg["base_label"],
                    "rbf_label": cfg["rbf_label"],
                    **{f"base_{int(m)}": base[i] for i, m in enumerate(MILESTONES)},
                    **{f"rbf_{int(m)}": rbf[i] for i, m in enumerate(MILESTONES)},
                    **{f"ratio_{int(m)}": ratio[i] for i, m in enumerate(MILESTONES)},
                    "n_hitos_rbf_mejor_10_90": int(np.sum(ratio[:-1] < 1.0)),
                    "mejor_ratio_10_90": float(np.min(ratio[:-1])),
                    "ratio_final_100": float(ratio[-1]),
                }
            )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(OUT_CSV)


if __name__ == "__main__":
    main()
