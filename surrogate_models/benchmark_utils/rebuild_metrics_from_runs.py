from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocesado_de_datos.utils.path_utils import escribir_json
from surrogate_models.benchmark_utils.benchmark_io import construir_payload_metricas, resumir_runs


DEFAULT_EXPERIMENT_ROOT = (
    ROOT
    / "results"
    / "cec"
    / "experimentos_mhs_ambos_cec2017_d10_tam50"
)

SUMMARY_SCRIPT_BY_PROTOCOL = {
    "acumulativo": ROOT / "surrogate_models" / "benchmark_utils" / "acumulativo" / "resumir_acumulativo.py",
    "no_acumulativo": ROOT / "surrogate_models" / "benchmark_utils" / "no_acumulativo" / "no_resumir_acumulativo.py",
}

SUMMARY_METRIC_KEYS = {
    "mae",
    "mae_std",
    "nmae",
    "nmae_std",
    "rmse",
    "rmse_std",
    "nrmse",
    "nrmse_std",
    "spearman",
    "spearman_std",
    "spearman_n_validas",
    "spearman_n_nan",
    "max_abs_error",
    "max_abs_error_std",
    "max_pct_error",
    "max_pct_error_std",
    "train_time_s",
    "train_time_s_std",
    "predict_time_s",
    "predict_time_s_std",
    "runs",
    "n_runs_evaluadas",
    "n_seeds_evaluadas",
    "n_train",
    "n_test",
}

RUN_NUMERIC_FIELDS = {
    "seed",
    "mae",
    "nmae",
    "rmse",
    "nrmse",
    "spearman",
    "max_abs_error",
    "max_pct_error",
    "n_train",
    "n_test",
    "batch_train",
    "batch_train_last",
    "train_pct_ini",
    "train_pct_fin",
    "eval_id_train_min",
    "eval_id_train_max",
    "eval_id_val_min",
    "eval_id_val_max",
    "train_time_s",
    "predict_time_s",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenera *_metricas.json a partir de los *_runs.csv ya existentes y "
            "actualiza los resumenes por funcion para estrategias acumulativa y/o no acumulativa."
        )
    )
    parser.add_argument(
        "--experiment-root",
        default=str(DEFAULT_EXPERIMENT_ROOT),
        help="Directorio raiz del experimento que contiene benchmarking/.",
    )
    parser.add_argument(
        "--protocols",
        nargs="*",
        choices=sorted(SUMMARY_SCRIPT_BY_PROTOCOL),
        default=sorted(SUMMARY_SCRIPT_BY_PROTOCOL),
        help="Protocolos a reprocesar.",
    )
    parser.add_argument(
        "--functions",
        nargs="*",
        default=None,
        help="Si se indica, restringe el reprocesado a estas funciones (ej. f1 f3).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Muestra lo que se reprocesaria sin escribir cambios.",
    )
    return parser.parse_args()


def read_runs_csv(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = []
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if key in RUN_NUMERIC_FIELDS and value is not None and value != "":
                    parsed[key] = float(value)
                else:
                    parsed[key] = value
            rows.append(parsed)
        return rows


def rebuild_metric_file(metric_path: Path, runs_csv_path: Path, *, dry_run: bool) -> bool:
    existing = json.loads(metric_path.read_text(encoding="utf-8"))
    runs = read_runs_csv(runs_csv_path)
    rebuilt = resumir_runs(runs)

    preserved = {key: value for key, value in existing.items() if key not in SUMMARY_METRIC_KEYS}
    rebuilt.update(preserved)
    payload = construir_payload_metricas(rebuilt)

    if dry_run:
        return payload != existing

    escribir_json(metric_path, payload)
    return payload != existing


def iter_metric_pairs(function_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for metric_path in sorted(function_dir.glob("*/*/*_metricas.json")):
        model_dir = metric_path.parent
        runs_csv_path = model_dir / f"{model_dir.name}_runs.csv"
        if runs_csv_path.is_file():
            pairs.append((metric_path, runs_csv_path))

    for metric_path in sorted(function_dir.glob("*/*/*/*_metricas.json")):
        batch_dir = metric_path.parent
        runs_csv_path = batch_dir / f"{batch_dir.parent.name}_runs.csv"
        if runs_csv_path.is_file():
            pairs.append((metric_path, runs_csv_path))

    return pairs


def refresh_function_summary(protocol: str, function_dir: Path, *, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] resumen {protocol}: {function_dir}")
        return

    script_path = SUMMARY_SCRIPT_BY_PROTOCOL[protocol]
    subprocess.run(
        [sys.executable, str(script_path), "--benchmark-dir", str(function_dir)],
        check=True,
    )


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.experiment_root).resolve()
    benchmarking_root = experiment_root / "benchmarking"
    total_updated = 0

    for protocol in args.protocols:
        protocol_root = benchmarking_root / protocol
        if not protocol_root.is_dir():
            print(f"[skip] no existe {protocol_root}")
            continue

        function_dirs = [path for path in sorted(protocol_root.iterdir()) if path.is_dir()]
        if args.functions is not None:
            allowed = set(args.functions)
            function_dirs = [path for path in function_dirs if path.name in allowed]

        for function_dir in function_dirs:
            metric_pairs = iter_metric_pairs(function_dir)
            updated_in_function = 0
            for metric_path, runs_csv_path in metric_pairs:
                changed = rebuild_metric_file(metric_path, runs_csv_path, dry_run=args.dry_run)
                if changed:
                    updated_in_function += 1
                    total_updated += 1
                    print(f"[update] {metric_path}")

            refresh_function_summary(protocol, function_dir, dry_run=args.dry_run)
            print(
                f"[done] {protocol}/{function_dir.name}: "
                f"{updated_in_function} metricas regeneradas"
            )

    print(f"[total] metricas regeneradas: {total_updated}")


if __name__ == "__main__":
    main()
