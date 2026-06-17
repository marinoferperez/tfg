"""
Genera tablas LaTeX de comparación de modelos desglosas por bloque temporal,
separadas por algoritmo (AGE, DE, SHADE).

Para cada combinación (función, bloque) se rankean los modelos por Spearman
(rango 1 = mejor). El rango medio se promedia sobre las funciones disponibles.
La tabla muestra una columna de rank por bloque más tiempos de entrenamiento
y predicción agregados sobre funciones y bloques.

Ficheros de salida:
  comparativa_modelos_bloques_age.tex
  comparativa_modelos_bloques_de.tex
  comparativa_modelos_bloques_shade.tex
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_BASE_DIR = (
    "results/cec/cec2017_d10_tam50/benchmarking"
    "/benchmarking_surrogates_offline_all/future_all"
)
DEFAULT_OUT_DIR = "memoria/tablas"

MODELOS = ["hgb", "lasso", "mlp", "random_forest", "rbf", "rsm", "svr", "xgboost"]
MODELOS_DISPLAY = {
    "hgb": "HGB",
    "lasso": "LASSO",
    "mlp": "MLP",
    "random_forest": "RF",
    "rbf": "RBF",
    "rsm": "RSM",
    "svr": "SVR",
    "xgboost": "XGBoost",
}
ALGORITMOS = ["age", "de", "shade"]
ALGO_LABELS = {"age": "AGE", "de": "DE", "shade": "SHADE"}
BLOQUES = ["1-20", "21-40", "41-60", "61-80"]
BLOQUE_LABELS = {"1-20": "20\\%", "21-40": "40\\%", "41-60": "60\\%", "61-80": "80\\%"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--prefix",
        default="comparativa_modelos_bloques",
        help="Prefijo de los ficheros de salida.",
    )
    return parser.parse_args()


def cargar_datos(base_dir: Path) -> pd.DataFrame:
    rows = []
    for modelo in MODELOS:
        base_est = base_dir / modelo / "no_acumulativo"
        if not base_est.exists():
            continue
        for funcion_dir in sorted(base_est.iterdir()):
            if not funcion_dir.is_dir():
                continue
            funcion = funcion_dir.name
            for algoritmo in ALGORITMOS:
                alg_dir = funcion_dir / algoritmo / modelo
                if not alg_dir.exists():
                    continue
                for bloque in BLOQUES:
                    json_path = alg_dir / bloque / f"{modelo}_metricas.json"
                    if not json_path.exists():
                        continue
                    data = json.loads(json_path.read_text(encoding="utf-8"))
                    rows.append(
                        {
                            "modelo": modelo,
                            "funcion": funcion,
                            "algoritmo": algoritmo,
                            "bloque": bloque,
                            "spearman": float(data.get("spearman") or 0),
                            "train_time_s": float(data.get("train_time_s") or 0),
                            "predict_time_s": float(data.get("predict_time_s") or 0),
                        }
                    )
    return pd.DataFrame(rows)


def calcular_rankings(df: pd.DataFrame, algoritmo: str) -> pd.DataFrame:
    """
    Para un algoritmo dado devuelve un DataFrame con:
      - índice: modelo
      - columna por bloque: rank medio sobre funciones (1=mejor Spearman)
      - train_time_s, predict_time_s: medios sobre funciones y bloques
    """
    df_alg = df[df["algoritmo"] == algoritmo].copy()

    # Redondeo a 4 decimales antes del ranking (criterio TACOLAB)
    df_alg["spearman"] = df_alg["spearman"].round(4)

    # Rank por (funcion, bloque) entre modelos
    df_alg["rank"] = df_alg.groupby(["funcion", "bloque"])["spearman"].rank(
        ascending=False, method="min"
    )

    # Rank medio por (modelo, bloque)
    rank_por_bloque = (
        df_alg.groupby(["modelo", "bloque"])["rank"]
        .mean()
        .unstack("bloque")
        .reindex(columns=BLOQUES)
    )

    # Tiempos medios por modelo (sobre funciones y bloques)
    tiempos = df_alg.groupby("modelo")[["train_time_s", "predict_time_s"]].mean()

    result = rank_por_bloque.join(tiempos)

    # Ordenar por rank medio global (promedio de los 4 bloques)
    result["rank_global"] = result[BLOQUES].mean(axis=1)
    result = result.sort_values("rank_global")

    return result


def _bold(s: str) -> str:
    return r"\textbf{" + s + r"}"


def _fmt_rank(val: float, best: float) -> str:
    s = f"{val:.4f}"
    return _bold(s) if abs(val - best) < 1e-9 else s


def _fmt_time(val: float, best: float) -> str:
    s = f"{val:.4f}"
    return _bold(s) if abs(val - best) < 1e-9 else s


def render_table(rank_df: pd.DataFrame, algo_label: str) -> str:
    n_bloques = len(BLOQUES)
    # col spec: modelo + 4 bloques + 2 tiempos
    col_spec = "l" + "r" * n_bloques + "|" + "rr"

    # Encabezado de bloques
    bloque_headers = " & ".join(
        f"\\textbf{{{BLOQUE_LABELS[b]}}}" for b in BLOQUES
    )
    header_line = (
        f"            \\textbf{{Modelo}} & {bloque_headers} "
        r"& \textbf{Entren.\ (s)} & \textbf{Pred.\ (s)} \\"
    )

    # Mejores valores por columna
    best_per_bloque = {b: rank_df[b].min() for b in BLOQUES}
    best_train = rank_df["train_time_s"].min()
    best_pred = rank_df["predict_time_s"].min()

    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        r"    \small",
        r"    \setlength{\tabcolsep}{4pt}",
        r"    \begin{adjustbox}{max width=\textwidth}",
        f"        \\begin{{tabular}}{{{col_spec}}}",
        r"            \toprule",
        # Subencabezado de bloques
        (
            f"            \\multicolumn{{1}}{{l}}{{}} & "
            f"\\multicolumn{{{n_bloques}}}{{c}}{{\\textbf{{Rank medio por bloque}}}} "
            r"& \multicolumn{2}{c}{\textbf{Tiempos medios}} \\"
        ),
        f"            \\cmidrule(lr){{2-{1+n_bloques}}} "
        f"\\cmidrule(lr){{{2+n_bloques}-{3+n_bloques}}}",
        header_line,
        r"            \midrule",
    ]

    for modelo, row in rank_df.iterrows():
        name = MODELOS_DISPLAY.get(modelo, modelo.upper())
        rank_cols = " & ".join(
            _fmt_rank(row[b], best_per_bloque[b]) for b in BLOQUES
        )
        lines.append(
            f"            {name} & {rank_cols} "
            f"& {_fmt_time(row['train_time_s'], best_train)} "
            f"& {_fmt_time(row['predict_time_s'], best_pred)} \\\\"
        )

    algo_lower = algo_label.lower()
    lines += [
        r"            \bottomrule",
        r"        \end{tabular}",
        r"    \end{adjustbox}",
        (
            f"    \\caption{{Competición de modelos para {algo_label} bajo la "
            r"estrategia no acumulativa, desglosada por bloque temporal. "
            r"Rank medio calculado sobre las funciones de prueba para cada "
            r"bloque de validación (1\,=\,mejor Spearman). "
            r"Tiempos agregados sobre funciones y bloques.}"
        ),
        f"    \\label{{tab:comparativa_modelos_bloques_{algo_lower}}}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cargando datos desde {base_dir} ...")
    df = cargar_datos(base_dir)
    print(f"  {len(df)} filas | modelos: {sorted(df['modelo'].unique())} | funciones: {len(df['funcion'].unique())}")

    for algo in ALGORITMOS:
        label = ALGO_LABELS[algo]
        rank_df = calcular_rankings(df, algo)
        tex = render_table(rank_df, label)
        out_path = out_dir / f"{args.prefix}_{algo}.tex"
        out_path.write_text(tex, encoding="utf-8")

        print(f"\n  {label}:")
        header = f"  {'Modelo':>10} " + " ".join(f"{'blk'+b:>8}" for b in BLOQUES) + f"  {'Train':>8} {'Pred':>8}"
        print(header)
        for modelo, row in rank_df.iterrows():
            ranks = " ".join(f"{row[b]:>8.2f}" for b in BLOQUES)
            print(f"  {MODELOS_DISPLAY.get(modelo,modelo):>10} {ranks}  {row['train_time_s']:>8.4f} {row['predict_time_s']:>8.4f}")
        print(f"  → {out_path}")


if __name__ == "__main__":
    main()
