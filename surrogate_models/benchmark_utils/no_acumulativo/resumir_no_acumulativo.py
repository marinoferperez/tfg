from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocesado_de_datos.utils.path_utils import (
    detectar_algoritmos_benchmark,
    resolver_directorio_existente,
)
from surrogate_models.benchmark_utils.benchmark_summary import (
    cargar_metricas_batch_algoritmo_generico,
    cargar_metricas_algoritmo_generico,
    generar_analisis_por_batch,
    generar_rankings_genericos,
    generar_resumenes_genericos,
    imprimir_resumen,
)

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Consolida los resultados del benchmark no_acumulativo para una funcion "
            "y genera rankings por Spearman."
        )
    )
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--algoritmos", nargs="*", default=None)
    parser.add_argument(
        "--rank-method",
        default="average",
        choices=["average", "min", "max", "dense", "first"],
    )
    parser.add_argument("--no-rankings", action="store_true")
    parser.add_argument("--no-json", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    benchmark_dir = resolver_directorio_existente(args.benchmark_dir, arg_name="benchmark_dir")
    algoritmos = args.algoritmos or detectar_algoritmos_benchmark(benchmark_dir)
    if not algoritmos:
        raise ValueError(f"No se encontraron directorios de benchmark con metricas en {benchmark_dir}")

    problema = benchmark_dir.name
    resumen_global = []
    resumen_por_batch = []

    for algoritmo in algoritmos:
        rows = cargar_metricas_algoritmo_generico(
            benchmark_dir,
            algoritmo,
            campos_extra={
                "protocol": lambda data: data.get("protocol"),
                "validation_ratio": lambda data: data.get("validation_ratio"),
                "n_seeds_evaluadas": lambda data: data.get("n_seeds_evaluadas"),
                "n_seeds_sin_casos_validacion": lambda data: data.get("n_seeds_sin_casos_validacion"),
                "convergencia_criterio": lambda data: data.get("convergencia_criterio"),
            },
        )
        if not rows:
            continue
        imprimir_resumen(rows, algoritmo)
        resumen_global.extend(rows)
        resumen_por_batch.extend(
            cargar_metricas_batch_algoritmo_generico(
                benchmark_dir,
                algoritmo,
                campos_extra={
                    "protocol": lambda data: data.get("protocol"),
                    "validation_ratio": lambda data: data.get("validation_ratio"),
                    "n_seeds_evaluadas": lambda data: data.get("n_seeds_evaluadas"),
                    "batch_train": lambda data: data.get("batch_train"),
                    "batch_train_last": lambda data: data.get("batch_train_last"),
                    "batch_label": lambda data: data.get("batch_label"),
                    "train_pct_ini": lambda data: data.get("train_pct_ini"),
                    "train_pct_fin": lambda data: data.get("train_pct_fin"),
                },
            )
        )

    if not resumen_global:
        raise ValueError(f"No se encontraron archivos *_metricas.json en {benchmark_dir}")

    generar_resumenes_genericos(
        resumen_global,
        benchmark_dir,
        problema,
        nombre_csv=f"resumen_benchmark_no_acumulativo_{problema}.csv",
        nombre_json=None if args.no_json else f"resumen_benchmark_no_acumulativo_{problema}.json",
        payload_extra={"protocol": "no_acumulativo"},
    )
    if not args.no_rankings:
        generar_rankings_genericos(resumen_global, benchmark_dir, problema, args.rank_method)
    generar_analisis_por_batch(
        resumen_por_batch,
        benchmark_dir,
        problema,
        prefijo_resumen="resumen_benchmark_no_acumulativo",
        payload_extra={"protocol": "no_acumulativo"},
        rank_method=args.rank_method,
        generar_rankings=not args.no_rankings,
        generar_json=not args.no_json,
    )

if __name__ == "__main__":
    main()
