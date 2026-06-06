import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocesado_de_datos.utils.path_utils import (
    detectar_algoritmos_benchmark,
    escribir_csv,
    escribir_json,
    listar_metricas_json_algoritmo,
    resolver_directorio_existente,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Consolida los resultados de benchmark_runner_cec.py y genera "
            "rankings por Spearman para una funcion concreta."
        )
    )
    parser.add_argument(
        "--benchmark-dir",
        required=True,
        help=(
            "Ruta al directorio benchmark_surrogates de una funcion concreta, "
            "por ejemplo .../f1/benchmark_surrogates."
        ),
    )
    parser.add_argument(
        "--algoritmos",
        nargs="*",
        default=None,
        help="Lista opcional de algoritmos a consolidar. Default: autodetectar.",
    )
    parser.add_argument(
        "--rank-method",
        default="average",
        choices=["average", "min", "max", "dense", "first"],
        help="Metodo de ranking de pandas. Default: average.",
    )
    parser.add_argument(
        "--no-rankings",
        action="store_true",
        help="Desactiva la generacion de ranking_detallado_*.csv y ranking_general_*.csv.",
    )
    return parser.parse_args()


def cargar_metricas_algoritmo(benchmark_dir, algoritmo):
    rows = []
    for ruta in listar_metricas_json_algoritmo(benchmark_dir, algoritmo):
        data = json.loads(ruta.read_text(encoding="utf-8"))
        row = {
            "algoritmo": algoritmo,
            "modelo": data["model"],
            "feature_mode": data["feature_mode"],
            "split_strategy": data["split_strategy"],
            "n_runs_evaluadas": data.get("n_runs_evaluadas"),
            "n_train": data["n_train"],
            "n_test": data["n_test"],
            "mae": data["mae"],
            "mae_std": data["mae_std"],
            "rmse": data["rmse"],
            "rmse_std": data["rmse_std"],
            "spearman": data["spearman"],
            "spearman_std": data["spearman_std"],
            "max_abs_error": data["max_abs_error"],
            "max_abs_error_std": data["max_abs_error_std"],
            "max_pct_error": data["max_pct_error"],
            "max_pct_error_std": data["max_pct_error_std"],
            "train_time_s": data["train_time_s"],
            "train_time_s_std": data["train_time_s_std"],
            "predict_time_s": data["predict_time_s"],
            "predict_time_s_std": data["predict_time_s_std"],
            "ruta_metricas": str(ruta),
        }
        rows.append(row)
    return rows


def ordenar_resumen(rows):
    return sorted(
        rows,
        key=lambda row: (
            -row["spearman"],
            row["rmse"],
            row["train_time_s"],
            row["modelo"],
        ),
    )


def imprimir_resumen(rows, algoritmo):
    print(f"[{algoritmo}]")
    for idx, row in enumerate(ordenar_resumen(rows), start=1):
        print(
            f"{idx}. {row['modelo']} | "
            f"Spearman={row['spearman']:.6f} | "
            f"RMSE={row['rmse']:.3f} | "
            f"train={row['train_time_s']:.6f}s | "
            f"pred={row['predict_time_s']:.6f}s"
        )


def construir_dataframe_rankings(rows, problema):
    df = pd.DataFrame(rows).copy()
    df["problema"] = problema
    return df


def compute_ranking(df, metric="spearman", rank_method="average"):
    group_cols = ["problema", "algoritmo"]
    df_rank = df.copy()
    df_rank["rank"] = (
        df_rank.groupby(group_cols)[metric]
        .rank(ascending=False, method=rank_method)
    )
    orden = df_rank.sort_values(group_cols + ["rank", "modelo"]).reset_index(drop=True)
    return orden[["problema", "algoritmo", "modelo", metric, "rank"]]


def ranking_medio(df_rank, metric):
    ranking = (
        df_rank.groupby("modelo")
        .agg(
            rank_medio=("rank", "mean"),
            rank_std=("rank", "std"),
            valor_medio=(metric, "mean"),
            valor_std=(metric, "std"),
        )
        .reset_index()
        .sort_values(["rank_medio", "valor_medio", "modelo"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    ranking["rank_std"] = ranking["rank_std"].fillna(0.0)
    ranking["valor_std"] = ranking["valor_std"].fillna(0.0)
    return ranking

def dataframe_to_json_records(df):
    return json.loads(df.to_json(orient="records"))


def generar_resumenes(rows, benchmark_dir, problema):
    df = pd.DataFrame(rows).copy()
    df = ordenar_resumen(df.to_dict(orient="records"))
    df = pd.DataFrame(df)

    csv_global = benchmark_dir / f"resumen_benchmark_{problema}.csv"
    json_global = benchmark_dir / f"resumen_benchmark_{problema}.json"
    escribir_csv(df, csv_global)
    escribir_json(
        json_global,
        {
            "problema": problema,
            "algoritmos": sorted(df["algoritmo"].unique().tolist()),
            "n_resultados": int(len(df)),
            "resultados": dataframe_to_json_records(df),
        },
    )


def generar_rankings(rows, benchmark_dir, problema, rank_method):
    df = construir_dataframe_rankings(rows, problema)
    for metric in ("spearman",):
        df_rank = compute_ranking(df, metric=metric, rank_method=rank_method)
        df_rank_medio = ranking_medio(df_rank, metric=metric)
        escribir_csv(df_rank, benchmark_dir / f"ranking_detallado_{metric}.csv")
        escribir_csv(df_rank_medio, benchmark_dir / f"ranking_general_{metric}.csv")
        print(f"[ranking {metric}]")
        print(df_rank_medio.to_string(index=False))


def main():
    args = parse_args()
    benchmark_dir = resolver_directorio_existente(args.benchmark_dir, arg_name="benchmark_dir")

    algoritmos = args.algoritmos or detectar_algoritmos_benchmark(benchmark_dir)
    if not algoritmos:
        raise ValueError(
            f"No se encontraron directorios de benchmark con metricas en {benchmark_dir}"
        )

    problema = benchmark_dir.parent.name
    resumen_global = []

    for algoritmo in algoritmos:
        rows = cargar_metricas_algoritmo(benchmark_dir, algoritmo)
        if not rows:
            continue

        imprimir_resumen(rows, algoritmo)
        resumen_global.extend(rows)

    if not resumen_global:
        raise ValueError(f"No se encontraron archivos *_metricas.json en {benchmark_dir}")

    generar_resumenes(
        rows=resumen_global,
        benchmark_dir=benchmark_dir,
        problema=problema,
    )

    if not args.no_rankings:
        generar_rankings(
            rows=resumen_global,
            benchmark_dir=benchmark_dir,
            problema=problema,
            rank_method=args.rank_method,
        )

if __name__ == "__main__":
    main()
