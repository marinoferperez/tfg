import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from surrogate_models.benchmark_plots import generar_boxplots_metricas_por_algoritmo
from preprocesado_de_datos.utils.path_utils import (
    clave_funcion,
    detectar_algoritmos_benchmark,
    detectar_funciones_experimento,
    escribir_csv,
    listar_metricas_json_algoritmo,
    resolver_directorio_existente,
)


METRICAS_PRINCIPALES = [
    "mae",
    "rmse",
    "spearman",
    "max_abs_error",
    "max_pct_error",
    "train_time_s",
    "predict_time_s",
]

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Consolida los benchmarks de todas las funciones de un experimento "
            "y genera resúmenes globales por función, por algoritmo y globales."
        )
    )
    parser.add_argument(
        "--experiment-dir",
        required=True,
        help=(
            "Ruta al directorio del experimento, por ejemplo "
            "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10."
        ),
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help=(
            "Directorio de salida. Default: <experiment-dir>/benchmark_surrogates_global."
        ),
    )
    parser.add_argument(
        "--funciones",
        nargs="*",
        default=None,
        help="Lista opcional de funciones a incluir. Acepta f1 o 1.",
    )
    parser.add_argument(
        "--algoritmos",
        nargs="*",
        default=None,
        help="Lista opcional de algoritmos a incluir. Default: autodetectar.",
    )
    parser.add_argument(
        "--rank-method",
        default="average",
        choices=["average", "min", "max", "dense", "first"],
        help="Metodo de ranking de pandas. Default: average.",
    )
    return parser.parse_args()

def cargar_metricas_experimento(experiment_dir, funciones, algoritmos=None):
    rows = []

    for funcion in funciones:
        benchmark_dir = experiment_dir / funcion / "benchmark_surrogates"
        algoritmos_funcion = algoritmos or detectar_algoritmos_benchmark(benchmark_dir)

        for algoritmo in algoritmos_funcion:
            for ruta in listar_metricas_json_algoritmo(benchmark_dir, algoritmo):
                data = json.loads(ruta.read_text(encoding="utf-8"))
                row = {
                    "funcion": funcion,
                    "algoritmo": algoritmo,
                    "modelo": data["model"],
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
                    "n_folds": data.get("n_folds"),
                    "n_train_media": data.get("n_train"),
                    "n_test_media": data.get("n_test"),
                }
                rows.append(row)

    if not rows:
        raise ValueError(
            f"No se encontraron archivos *_metricas.json en {experiment_dir}"
        )

    df = pd.DataFrame(rows)
    df = ordenar_por_funcion(df, ["funcion", "algoritmo", "modelo"])
    return df


def ordenar_por_funcion(df, columnas):
    if "funcion" not in df.columns:
        return df.sort_values(columnas).reset_index(drop=True)
    copia = df.copy()
    copia["_funcion_orden"] = copia["funcion"].map(clave_funcion)
    orden = ["_funcion_orden"] + [col for col in columnas if col != "funcion"]
    copia = copia.sort_values(orden).drop(columns="_funcion_orden").reset_index(drop=True)
    return copia

def agregar_metricas(df, group_cols, std_label):
    agg = {}
    for metrica in METRICAS_PRINCIPALES:
        agg[f"{metrica}_media"] = (metrica, "mean")
        agg[f"{metrica}_{std_label}"] = (metrica, "std")

    resumen = (
        df.groupby(group_cols)
        .agg(**agg)
        .reset_index()
        .sort_values(group_cols)
        .reset_index(drop=True)
    )
    columnas_std = [col for col in resumen.columns if col.endswith(f"_{std_label}")]
    if columnas_std:
        resumen[columnas_std] = resumen[columnas_std].fillna(0.0)
    if "funcion" in resumen.columns:
        resumen = ordenar_por_funcion(resumen, group_cols)
    return resumen


def compute_case_ranking(df, metric, rank_method):
    group_cols = ["funcion", "algoritmo"]
    ranking = df[group_cols + ["modelo", metric]].copy()
    ranking["rank"] = (
        ranking.groupby(group_cols)[metric]
        .rank(ascending=False, method=rank_method)
    )
    ranking = ordenar_por_funcion(ranking, group_cols + ["rank", "modelo"])
    return ranking


def ranking_medio_por_algoritmo(df_rank, metric):
    ranking = (
        df_rank.groupby(["algoritmo", "modelo"])
        .agg(
            rank_medio=("rank", "mean"),
            rank_std=("rank", "std"),
            **{
                f"{metric}_medio": (metric, "mean"),
                f"{metric}_std": (metric, "std"),
            },
        )
        .reset_index()
        .sort_values(["algoritmo", "rank_medio", f"{metric}_medio", "modelo"], ascending=[True, True, False, True])
        .reset_index(drop=True)
    )
    ranking["rank_std"] = ranking["rank_std"].fillna(0.0)
    ranking[f"{metric}_std"] = ranking[f"{metric}_std"].fillna(0.0)
    return ranking


def ranking_medio_global(df_rank, metric):
    ranking = (
        df_rank.groupby("modelo")
        .agg(
            rank_medio=("rank", "mean"),
            rank_std=("rank", "std"),
            **{
                f"{metric}_medio": (metric, "mean"),
                f"{metric}_std": (metric, "std"),
            },
        )
        .reset_index()
        .sort_values(["rank_medio", f"{metric}_medio", "modelo"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    ranking["rank_std"] = ranking["rank_std"].fillna(0.0)
    ranking[f"{metric}_std"] = ranking[f"{metric}_std"].fillna(0.0)
    return ranking


def construir_tabla_b_ranking_por_funcion(df, rank_method):
    ranking_spearman = compute_case_ranking(df, metric="spearman", rank_method=rank_method)
    ranking_spearman = ranking_spearman.rename(columns={"rank": "rank_spearman"})

    tabla_b = (
        df[["funcion", "algoritmo", "modelo", "spearman"]]
        .merge(
            ranking_spearman[["funcion", "algoritmo", "modelo", "rank_spearman"]],
            on=["funcion", "algoritmo", "modelo"],
            how="left",
        )
    )
    return ordenar_por_funcion(
        tabla_b,
        ["funcion", "algoritmo", "rank_spearman", "modelo"],
    )


def construir_tablas_ranking(df, outdir, rank_method):
    for metric in ("spearman",):
        detalle = compute_case_ranking(df, metric=metric, rank_method=rank_method)
        escribir_csv(
            ranking_medio_por_algoritmo(detalle, metric=metric),
            Path(outdir) / f"ranking_medio_{metric}_por_algoritmo.csv",
        )
        escribir_csv(
            ranking_medio_global(detalle, metric=metric),
            Path(outdir) / f"ranking_medio_{metric}_global.csv",
        )


def generar_boxplots_metricas_experimento(df, outdir):
    return generar_boxplots_metricas_por_algoritmo(
        df,
        Path(outdir) / "boxplots_metricas_por_funcion",
        contexto_titulo="distribucion entre funciones",
    )


def main():
    args = parse_args()
    experiment_dir = resolver_directorio_existente(args.experiment_dir, arg_name="experiment_dir")

    outdir = Path(args.outdir).resolve() if args.outdir else experiment_dir / "benchmark_surrogates_global"
    funciones = detectar_funciones_experimento(
        experiment_dir,
        funciones=args.funciones,
        required_subdir="benchmark_surrogates",
    )
    df = cargar_metricas_experimento(
        experiment_dir=experiment_dir,
        funciones=funciones,
        algoritmos=args.algoritmos,
    )

    escribir_csv(df, outdir / "resumen_por_funcion.csv")
    escribir_csv(
        agregar_metricas(df, ["algoritmo", "modelo"], std_label="std_entre_funciones"),
        outdir / "resumen_por_algoritmo.csv",
    )
    escribir_csv(
        construir_tabla_b_ranking_por_funcion(df, rank_method=args.rank_method),
        outdir / "ranking_por_funcion.csv",
    )

    construir_tablas_ranking(df, outdir=outdir, rank_method=args.rank_method)
    generar_boxplots_metricas_experimento(df, outdir=outdir)

    print(f"Funciones agregadas: {', '.join(funciones)}")
    print(f"Salida global en: {outdir}")


if __name__ == "__main__":
    main()
