from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
cache_dir = ROOT / "tmp" / ".mplconfig"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from surrogate_models.benchmark_utils.batches_eval_splitter import construir_casos_no_acumulativos
from surrogate_models.benchmark_utils.evaluacion_offline import ejecutar_benchmark_temporal


def parse_args():
    parser = argparse.ArgumentParser(
        description="Genera plots de rango real vs predicho por batch para no_acumulativo."
    )
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--funcion", default=None)
    parser.add_argument("--algoritmo", default=None, choices=["age", "de", None])
    parser.add_argument("--model", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Regenera los PNG solo a partir de los CSV ya existentes; no reconstruye sample_errors.",
    )
    return parser.parse_args()


def resolver_metricas_modelo(benchmark_root: Path, funcion: str | None, algoritmo: str | None, model: str | None):
    benchmark_root = benchmark_root.resolve()
    if not benchmark_root.exists():
        raise FileNotFoundError(f"No existe benchmark_root: {benchmark_root}")

    candidatos = sorted(benchmark_root.glob("f*/ */ */ *_metricas.json".replace(" ", "")))
    # Si benchmark_root ya apunta a una funcion concreta
    if not candidatos:
        candidatos = sorted(benchmark_root.glob("*/*_metricas.json"))

    metricas = []
    for ruta in candidatos:
        if ruta.parent.name.startswith(("1-", "21-", "41-", "61-")):
            continue
        if ruta.parent.parent.name == "por_batch":
            continue
        try:
            algo = ruta.parent.parent.name
            func = ruta.parent.parent.parent.name
            modelo = ruta.parent.name
        except IndexError:
            continue
        if funcion and func != funcion:
            continue
        if algoritmo and algo != algoritmo:
            continue
        if model and modelo != model:
            continue
        metricas.append(ruta)
    return metricas


def cargar_metricas(ruta_metricas: Path):
    with ruta_metricas.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def agrupar_errores_por_batch(sample_errors):
    agrupados = defaultdict(lambda: {"y_true": [], "y_pred": []})
    for fila in sample_errors:
        batch_label = str(fila["batch_label"])
        agrupados[batch_label]["y_true"].append(float(fila["y_true"]))
        agrupados[batch_label]["y_pred"].append(float(fila["y_pred"]))
    return dict(agrupados)


def resumir_rangos_por_batch(sample_errors):
    filas = []
    agrupados = agrupar_errores_por_batch(sample_errors)
    orden = sorted(agrupados.keys(), key=lambda s: int(str(s).split("-")[0]))
    for batch_label in orden:
        grupo = agrupados[batch_label]
        y_true = np.asarray(grupo["y_true"], dtype=float)
        y_pred = np.asarray(grupo["y_pred"], dtype=float)
        real_q = np.percentile(y_true, [0, 5, 50, 95, 100])
        pred_q = np.percentile(y_pred, [0, 5, 50, 95, 100])
        filas.append(
            {
                "batch_label": batch_label,
                "n_samples": int(y_true.size),
                "real_min": float(real_q[0]),
                "real_p5": float(real_q[1]),
                "real_p50": float(real_q[2]),
                "real_p95": float(real_q[3]),
                "real_max": float(real_q[4]),
                "pred_min": float(pred_q[0]),
                "pred_p5": float(pred_q[1]),
                "pred_p50": float(pred_q[2]),
                "pred_p95": float(pred_q[3]),
                "pred_max": float(pred_q[4]),
            }
        )
    return filas


def escribir_csv(ruta_csv: Path, filas):
    ruta_csv.parent.mkdir(parents=True, exist_ok=True)
    if not filas:
        ruta_csv.write_text("", encoding="utf-8")
        return
    with ruta_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        writer.writeheader()
        writer.writerows(filas)


def cargar_csv_resumen(ruta_csv: Path):
    if not ruta_csv.exists():
        return None
    with ruta_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        filas = []
        for fila in reader:
            filas.append(
                {
                    "batch_label": str(fila["batch_label"]),
                    "n_samples": int(float(fila["n_samples"])),
                    "real_min": float(fila["real_min"]),
                    "real_p5": float(fila["real_p5"]),
                    "real_p50": float(fila["real_p50"]),
                    "real_p95": float(fila["real_p95"]),
                    "real_max": float(fila["real_max"]),
                    "pred_min": float(fila["pred_min"]),
                    "pred_p5": float(fila["pred_p5"]),
                    "pred_p50": float(fila["pred_p50"]),
                    "pred_p95": float(fila["pred_p95"]),
                    "pred_max": float(fila["pred_max"]),
                }
            )
    return filas


def usar_escala_log(filas):
    valores = []
    for fila in filas:
        valores.extend(
            [
                fila["real_p5"],
                fila["real_p95"],
                fila["pred_p5"],
                fila["pred_p95"],
            ]
        )
    arr = np.asarray(valores, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or np.min(arr) <= 0:
        return False
    return (np.max(arr) / np.min(arr)) >= 1e3


def generar_plot_rangos(ruta_png: Path, filas, *, funcion: str, algoritmo: str, modelo: str):
    ruta_png.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(filas), dtype=float)
    labels = [fila["batch_label"] for fila in filas]

    real_p5 = np.asarray([fila["real_p5"] for fila in filas], dtype=float)
    real_p50 = np.asarray([fila["real_p50"] for fila in filas], dtype=float)
    real_p95 = np.asarray([fila["real_p95"] for fila in filas], dtype=float)
    pred_p5 = np.asarray([fila["pred_p5"] for fila in filas], dtype=float)
    pred_p50 = np.asarray([fila["pred_p50"] for fila in filas], dtype=float)
    pred_p95 = np.asarray([fila["pred_p95"] for fila in filas], dtype=float)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    x_real = x - 0.14
    x_pred = x + 0.14

    ax.errorbar(
        x_real,
        real_p50,
        yerr=np.vstack([real_p50 - real_p5, real_p95 - real_p50]),
        fmt="none",
        color="#4e79a7",
        elinewidth=3.4,
        capsize=4.5,
        capthick=2.4,
        alpha=0.95,
        zorder=4,
    )

    ax.errorbar(
        x_pred,
        pred_p50,
        yerr=np.vstack([pred_p50 - pred_p5, pred_p95 - pred_p50]),
        fmt="none",
        color="#f28e2b",
        elinewidth=3.4,
        capsize=4.5,
        capthick=2.4,
        alpha=0.95,
        zorder=4,
    )

    ax.plot(
        x_real,
        real_p50,
        linestyle="None",
        marker="_",
        color="black",
        markersize=7,
        markeredgewidth=1.0,
        zorder=5,
        label="Mediana real",
    )
    ax.plot(
        x_pred,
        pred_p50,
        linestyle="None",
        marker="_",
        color="black",
        markersize=7,
        markeredgewidth=1.0,
        zorder=5,
        label="Mediana predicha",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Batch de entrenamiento (validado en muestras futuras)")
    ax.set_ylabel("Fitness")
    ax.set_title(
        f"Rango efectivo real vs predicho por batch (p5-p95 y mediana) "
        f"para {funcion} con {algoritmo.upper()} y {modelo}"
    )
    ax.grid(alpha=0.22)
    if usar_escala_log(filas):
        ax.set_yscale("log")
    legend_handles = [
        Line2D([0], [0], color="#4e79a7", linewidth=3.4, label="Rango real (p5–p95)"),
        Line2D([0], [0], marker="_", color="black", linestyle="None", markersize=8, markeredgewidth=1.0, label="Mediana real"),
        Line2D([0], [0], color="#f28e2b", linewidth=3.4, label="Rango predicho (p5–p95)"),
        Line2D([0], [0], marker="_", color="black", linestyle="None", markersize=8, markeredgewidth=1.0, label="Mediana predicha"),
    ]
    ax.legend(handles=legend_handles, loc="best")
    fig.tight_layout()
    fig.savefig(ruta_png, dpi=180)
    plt.close(fig)


def reconstruir_errores_desde_metricas(metricas):
    resultado = ejecutar_benchmark_temporal(
        dataset_paths=metricas["datasets"],
        funcion=metricas["funcion"],
        algoritmo=metricas["algoritmo"],
        model_name=metricas["model"],
        model_kwargs=metricas.get("model_params") or {},
        constructor_casos=construir_casos_no_acumulativos,
        protocol="no_acumulativo",
        split_strategy=metricas["split_strategy"],
        random_state=int(metricas.get("random_state", 42)),
        collect_sample_errors=True,
    )
    return resultado["sample_errors"]


def main():
    args = parse_args()
    benchmark_root = Path(args.benchmark_root)
    rutas_metricas = resolver_metricas_modelo(
        benchmark_root,
        funcion=args.funcion,
        algoritmo=args.algoritmo,
        model=args.model,
    )
    if not rutas_metricas:
        raise RuntimeError("No se encontraron metricas de modelo para generar plots.")

    for ruta_metricas in rutas_metricas:
        metricas = cargar_metricas(ruta_metricas)
        model_dir = ruta_metricas.parent
        ruta_csv = model_dir / f"{metricas['model']}_rango_predicho_vs_real_por_batch.csv"
        ruta_png = model_dir / f"{metricas['model']}_rango_predicho_vs_real_por_batch.png"
        if not args.force and ruta_png.exists():
            print(f"[skip] {metricas['funcion']} {metricas['algoritmo']} {metricas['model']}")
            continue

        filas = cargar_csv_resumen(ruta_csv)
        if filas is None:
            if args.csv_only:
                print(f"[skip csv-only] {metricas['funcion']} {metricas['algoritmo']} {metricas['model']}")
                continue
            sample_errors = reconstruir_errores_desde_metricas(metricas)
            filas = resumir_rangos_por_batch(sample_errors)
            escribir_csv(ruta_csv, filas)

        generar_plot_rangos(
            ruta_png,
            filas,
            funcion=metricas["funcion"],
            algoritmo=metricas["algoritmo"],
            modelo=metricas["model"],
        )
        print(f"[{metricas['funcion']}] {metricas['algoritmo']} {metricas['model']} -> {ruta_png}")


if __name__ == "__main__":
    main()
