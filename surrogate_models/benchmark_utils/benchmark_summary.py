from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from preprocesado_de_datos.utils.path_utils import escribir_csv, escribir_json, listar_metricas_json_algoritmo
from surrogate_models.benchmark_utils.benchmark_paths import construir_nombre_batch


def cargar_metricas_algoritmo_generico(benchmark_dir, algoritmo, *, campos_extra=None):
    rows = []
    campos_extra = campos_extra or {}
    for ruta in listar_metricas_json_algoritmo(benchmark_dir, algoritmo):
        data = json.loads(ruta.read_text(encoding="utf-8"))
        rows.append(_construir_row_metricas(data, ruta, algoritmo, campos_extra))
    return rows


def _construir_row_metricas(data, ruta, algoritmo, campos_extra=None):
    row = {
        "algoritmo": algoritmo,
        "modelo": data["model"],
        "feature_mode": data["feature_mode"],
        "split_strategy": data["split_strategy"],
        "n_runs_evaluadas": data.get("n_runs_evaluadas"),
        "n_train": data.get("n_train"),
        "n_test": data.get("n_test"),
        "mae": data["mae"],
        "mae_std": data["mae_std"],
        "nmae": data["nmae"],
        "nmae_std": data["nmae_std"],
        "rmse": data["rmse"],
        "rmse_std": data["rmse_std"],
        "nrmse": data["nrmse"],
        "nrmse_std": data["nrmse_std"],
        "spearman": data["spearman"],
        "spearman_std": data["spearman_std"],
        "spearman_n_validas": data.get("spearman_n_validas"),
        "spearman_n_nan": data.get("spearman_n_nan"),
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
    for campo, extractor in (campos_extra or {}).items():
        row[campo] = extractor(data)
    return row


def cargar_metricas_batch_algoritmo_generico(benchmark_dir, algoritmo, *, campos_extra=None):
    rows = []
    base = Path(benchmark_dir)
    algoritmo_dir = base / algoritmo
    if not algoritmo_dir.is_dir():
        return rows

    for ruta in sorted(algoritmo_dir.glob("*/*/*_metricas.json")):
        data = json.loads(ruta.read_text(encoding="utf-8"))
        rows.append(_construir_row_metricas(data, ruta, algoritmo, campos_extra))
    return rows


def ordenar_resumen(rows):
    return sorted(
        rows,
        key=lambda row: (
            -row["spearman"],
            row["rmse"],
            row["mae"],
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
            f"RMSE={row['rmse']:.6f} | "
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


def _safe_int_from_series(df, column_name):
    if column_name not in df.columns:
        return None
    value = df[column_name].iloc[0]
    if pd.isna(value):
        return None
    return int(value)


def generar_resumenes_genericos(
    rows,
    benchmark_dir,
    problema,
    *,
    nombre_csv,
    nombre_json=None,
    payload_extra=None,
):
    df = pd.DataFrame(rows).copy()
    df = ordenar_resumen(df.to_dict(orient="records"))
    df = pd.DataFrame(df)
    df_export = df.copy()
    if "batch_label" in df_export.columns and "train_pct_ini" in df_export.columns and "train_pct_fin" in df_export.columns:
        df_export = df_export.drop(columns=["batch_label"])
    escribir_csv(df_export, benchmark_dir / nombre_csv)
    if nombre_json is None:
        return

    payload = {
        "problema": problema,
        "algoritmos": sorted(df_export["algoritmo"].unique().tolist()),
        "n_resultados": int(len(df_export)),
        "resultados": dataframe_to_json_records(df_export),
    }
    if payload_extra:
        payload.update(payload_extra)
    escribir_json(benchmark_dir / nombre_json, payload)


def generar_rankings_genericos(rows, benchmark_dir, problema, rank_method):
    df = construir_dataframe_rankings(rows, problema)
    for metric in ("spearman",):
        df_rank = compute_ranking(df, metric=metric, rank_method=rank_method)
        df_rank_medio = ranking_medio(df_rank, metric=metric)
        escribir_csv(df_rank, benchmark_dir / f"ranking_detallado_{metric}.csv")
        escribir_csv(df_rank_medio, benchmark_dir / f"ranking_general_{metric}.csv")
        print(f"[ranking {metric}]")
        print(df_rank_medio.to_string(index=False))


def generar_analisis_por_batch(
    rows,
    benchmark_dir,
    problema,
    *,
    prefijo_resumen,
    payload_extra=None,
    rank_method="average",
    generar_rankings=True,
    generar_json=True,
):
    if not rows:
        return

    df = pd.DataFrame(rows).copy()
    if "batch_label" not in df.columns:
        return

    por_batch_dir = Path(benchmark_dir) / "por_batch"
    for batch_label, df_batch in df.groupby("batch_label", sort=False):
        train_pct_ini = _safe_int_from_series(df_batch, "train_pct_ini")
        train_pct_fin = _safe_int_from_series(df_batch, "train_pct_fin")
        batch_dir_name = (
            construir_nombre_batch(train_pct_ini, train_pct_fin)
            if train_pct_ini is not None and train_pct_fin is not None
            else str(batch_label)
        )
        batch_dir = por_batch_dir / batch_dir_name
        rows_batch = df_batch.to_dict(orient="records")
        generar_resumenes_genericos(
            rows_batch,
            batch_dir,
            problema,
            nombre_csv=f"{prefijo_resumen}_{problema}.csv",
            nombre_json=f"{prefijo_resumen}_{problema}.json" if generar_json else None,
            payload_extra={
                **(payload_extra or {}),
                "batch_label": str(batch_label),
                "batch_train": _safe_int_from_series(df_batch, "batch_train"),
                "batch_train_last": _safe_int_from_series(df_batch, "batch_train_last"),
                "train_pct_ini": train_pct_ini,
                "train_pct_fin": train_pct_fin,
            },
        )
        if generar_rankings:
            generar_rankings_genericos(rows_batch, batch_dir, problema, rank_method)
