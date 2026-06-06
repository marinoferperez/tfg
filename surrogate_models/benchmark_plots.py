from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICAS_BOXPLOT = {
    "spearman": {
        "titulo": "Spearman",
        "ylabel": "Spearman",
        "ascending": False,
    },
    "rmse": {
        "titulo": "RMSE",
        "ylabel": "RMSE",
        "ascending": True,
    },
    "nrmse": {
        "titulo": "nRMSE",
        "ylabel": "nRMSE",
        "ascending": True,
    },
    "mae": {
        "titulo": "MAE",
        "ylabel": "MAE",
        "ascending": True,
    },
    "nmae": {
        "titulo": "nMAE",
        "ylabel": "nMAE",
        "ascending": True,
    },
    "max_abs_error": {
        "titulo": "Max Absolute Error",
        "ylabel": "Error absoluto maximo",
        "ascending": True,
    },
    "max_pct_error": {
        "titulo": "Max Percentage Error",
        "ylabel": "Error porcentual maximo",
        "ascending": True,
    },
    "train_time_s": {
        "titulo": "Training Time",
        "ylabel": "Tiempo de entrenamiento (s)",
        "ascending": True,
    },
    "predict_time_s": {
        "titulo": "Inference Time",
        "ylabel": "Tiempo de inferencia (s)",
        "ascending": True,
    },
}


def _ordenar_modelos(df_metrica, *, ascending):
    resumen = (
        df_metrica.groupby("modelo", as_index=False)[["valor"]]
        .median()
        .sort_values(["valor", "modelo"], ascending=[ascending, True])
    )
    return resumen["modelo"].tolist()


def _estilizar_boxplot(bp):
    for box in bp["boxes"]:
        box.set(facecolor="#cfe8ff", edgecolor="#34699a", linewidth=1.2)
    for whisker in bp["whiskers"]:
        whisker.set(color="#34699a", linewidth=1.1)
    for cap in bp["caps"]:
        cap.set(color="#34699a", linewidth=1.1)
    for median in bp["medians"]:
        median.set(color="#111111", linewidth=1.5)
    for flier in bp["fliers"]:
        flier.set(
            marker="o",
            markersize=4,
            markerfacecolor="#e63946",
            markeredgecolor="#e63946",
            alpha=0.65,
        )
    if "means" in bp:
        for mean in bp["means"]:
            mean.set(
                marker="D",
                markersize=5,
                markerfacecolor="#f4a261",
                markeredgecolor="#8d5524",
            )


def generar_boxplots_metricas_por_algoritmo(df, outdir, *, contexto_titulo):
    df = pd.DataFrame(df).copy()
    if df.empty:
        return []

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rutas = []

    for algoritmo, df_alg in df.groupby("algoritmo", sort=True):
        algoritmo_dir = outdir / algoritmo
        algoritmo_dir.mkdir(parents=True, exist_ok=True)

        for metrica, meta in METRICAS_BOXPLOT.items():
            if metrica not in df_alg.columns:
                continue

            df_metrica = df_alg[["modelo", metrica]].copy()
            df_metrica = df_metrica.rename(columns={metrica: "valor"})
            df_metrica["valor"] = pd.to_numeric(df_metrica["valor"], errors="coerce")
            df_metrica = df_metrica.replace([np.inf, -np.inf], np.nan).dropna(subset=["valor"])
            if df_metrica.empty:
                continue

            modelos_ordenados = _ordenar_modelos(df_metrica, ascending=meta["ascending"])
            series = [
                df_metrica.loc[df_metrica["modelo"] == modelo, "valor"].to_numpy(dtype=float)
                for modelo in modelos_ordenados
            ]

            if not series:
                continue

            fig_width = max(8.0, 1.15 * len(modelos_ordenados) + 2.5)
            fig, ax = plt.subplots(figsize=(fig_width, 5.4))
            try:
                bp = ax.boxplot(
                    series,
                    tick_labels=modelos_ordenados,
                    patch_artist=True,
                    showmeans=True,
                )
            except TypeError:
                bp = ax.boxplot(
                    series,
                    labels=modelos_ordenados,
                    patch_artist=True,
                    showmeans=True,
                )

            _estilizar_boxplot(bp)
            ax.set_title(f"{meta['titulo']} por modelo surrogate ({algoritmo.upper()}, {contexto_titulo})")
            ax.set_xlabel("Modelo surrogate")
            ax.set_ylabel(meta["ylabel"])
            ax.grid(axis="y", alpha=0.25, linestyle="--")
            ax.tick_params(axis="x", rotation=35)

            fig.tight_layout()
            outpath = algoritmo_dir / f"boxplot_{metrica}.png"
            fig.savefig(outpath, dpi=180, bbox_inches="tight")
            plt.close(fig)
            rutas.append(outpath)

    return rutas
