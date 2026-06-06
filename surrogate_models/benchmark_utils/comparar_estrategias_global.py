from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_BASE_DIR = (
    "results/cec/cec2017_d10_tam50_reinicio_seleccionado/"
    "benchmarking_surrogates_offline_next/future_next"
)
DEFAULT_FUNCIONES = ("f1", "f4", "f10", "f12", "f18", "f22", "f29")
DEFAULT_ALGORITMOS = ("age", "de", "shade")
DEFAULT_MODELOS = (
    "lasso",
    "rsm",
    "mlp",
    "rbf",
    "random_forest",
    "hgb",
    "xgboost",
    "svr",
)
PROTOCOLOS = ("no_acumulativo", "acumulativo")
BLOQUES_POR_PROTOCOLO = {
    "no_acumulativo": ("1-20", "21-40", "41-60", "61-80"),
    "acumulativo": ("1-20", "1-40", "1-60", "1-80"),
}
METRICAS = (
    "mae",
    "nmae",
    "rmse",
    "nrmse",
    "spearman",
    "train_time_s",
    "predict_time_s",
)
METRICAS_LATEX = ("spearman", "nmae", "nrmse", "train_time_s")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compara globalmente las estrategias offline acumulativa y no acumulativa "
            "a partir de los *_metricas.json temporales."
        )
    )
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--funciones", nargs="*", default=list(DEFAULT_FUNCIONES))
    parser.add_argument("--algoritmos", nargs="*", default=list(DEFAULT_ALGORITMOS))
    parser.add_argument("--modelos", nargs="*", default=list(DEFAULT_MODELOS))
    parser.add_argument(
        "--exclude-models",
        nargs="*",
        default=[],
        help="Modelos que se excluyen explicitamente de la comparacion agregada.",
    )
    parser.add_argument("--no-latex", action="store_true")
    return parser.parse_args()


def cargar_metricas(base_dir: Path, modelos, protocolos, funciones, algoritmos):
    rows = []
    for modelo in modelos:
        for protocolo in protocolos:
            for funcion in funciones:
                for algoritmo in algoritmos:
                    patron = (
                        f"{modelo}/{protocolo}/{funcion}/{algoritmo}/"
                        f"{modelo}/*/{modelo}_metricas.json"
                    )
                    for ruta in sorted(base_dir.glob(patron)):
                        data = json.loads(ruta.read_text(encoding="utf-8"))
                        batch_dir = ruta.parent.name
                        row = {
                            "modelo": modelo,
                            "estrategia": protocolo,
                            "funcion": funcion,
                            "algoritmo": algoritmo,
                            "bloque": batch_dir,
                            "ruta_metricas": str(ruta),
                            "n_seeds_evaluadas": data.get("n_seeds_evaluadas"),
                            "n_runs_evaluadas": data.get("n_runs_evaluadas"),
                            "feature_mode": data.get("feature_mode"),
                            "future_validation": data.get("future_validation"),
                            "split_strategy": data.get("split_strategy"),
                        }
                        for metrica in METRICAS:
                            row[metrica] = data.get(metrica)
                            row[f"{metrica}_std"] = data.get(f"{metrica}_std")
                        rows.append(row)
    return pd.DataFrame(rows)


def expected_combos(modelos, funciones, algoritmos, estrategia=None):
    if estrategia is not None:
        return len(modelos) * len(funciones) * len(algoritmos) * len(BLOQUES_POR_PROTOCOLO[estrategia])
    return sum(expected_combos(modelos, funciones, algoritmos, item) for item in PROTOCOLOS)


def resumir(df: pd.DataFrame, group_cols, modelos, funciones, algoritmos):
    agg = {
        "n_metricas": ("ruta_metricas", "count"),
        "n_runs_total": ("n_runs_evaluadas", "sum"),
        "n_seeds_min": ("n_seeds_evaluadas", "min"),
        "n_seeds_max": ("n_seeds_evaluadas", "max"),
    }
    for metrica in METRICAS:
        agg[f"{metrica}_mean"] = (metrica, "mean")
        agg[f"{metrica}_std"] = (metrica, "std")

    resumen = df.groupby(group_cols, dropna=False).agg(**agg).reset_index()
    resumen = resumen.fillna({f"{metrica}_std": 0.0 for metrica in METRICAS})

    def calcular_esperadas(row):
        estrategia = row["estrategia"] if "estrategia" in row else None
        modelos_row = [row["modelo"]] if "modelo" in row else modelos
        if estrategia is None:
            return expected_combos(modelos_row, funciones, algoritmos)
        return expected_combos(modelos_row, funciones, algoritmos, estrategia)

    resumen["n_metricas_esperadas"] = resumen.apply(calcular_esperadas, axis=1)
    resumen["cobertura_pct"] = 100.0 * resumen["n_metricas"] / resumen["n_metricas_esperadas"]
    return resumen


def comparar_modelos(resumen_modelo: pd.DataFrame):
    rows = []
    for modelo, df_modelo in resumen_modelo.groupby("modelo"):
        por_estrategia = {row["estrategia"]: row for _, row in df_modelo.iterrows()}
        if not all(item in por_estrategia for item in PROTOCOLOS):
            continue
        no_acum = por_estrategia["no_acumulativo"]
        acum = por_estrategia["acumulativo"]
        rows.append(
            {
                "modelo": modelo,
                "spearman_no_acumulativo": no_acum["spearman_mean"],
                "spearman_acumulativo": acum["spearman_mean"],
                "delta_spearman_acum_menos_no_acum": acum["spearman_mean"] - no_acum["spearman_mean"],
                "nmae_no_acumulativo": no_acum["nmae_mean"],
                "nmae_acumulativo": acum["nmae_mean"],
                "delta_nmae_acum_menos_no_acum": acum["nmae_mean"] - no_acum["nmae_mean"],
                "nrmse_no_acumulativo": no_acum["nrmse_mean"],
                "nrmse_acumulativo": acum["nrmse_mean"],
                "delta_nrmse_acum_menos_no_acum": acum["nrmse_mean"] - no_acum["nrmse_mean"],
                "train_time_s_no_acumulativo": no_acum["train_time_s_mean"],
                "train_time_s_acumulativo": acum["train_time_s_mean"],
                "delta_train_time_s_acum_menos_no_acum": acum["train_time_s_mean"] - no_acum["train_time_s_mean"],
                "estrategia_ganadora_spearman": (
                    "acumulativo"
                    if acum["spearman_mean"] > no_acum["spearman_mean"]
                    else "no_acumulativo"
                ),
                "estrategia_ganadora_nmae": (
                    "acumulativo"
                    if acum["nmae_mean"] < no_acum["nmae_mean"]
                    else "no_acumulativo"
                ),
                "cobertura_no_acumulativo_pct": no_acum["cobertura_pct"],
                "cobertura_acumulativo_pct": acum["cobertura_pct"],
            }
        )
    return pd.DataFrame(rows)


def fmt_float(value, digits=4):
    if pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


def latex_escape(value):
    return str(value).replace("_", r"\_")


def escribir_tabla_latex_modelo(resumen_modelo: pd.DataFrame, out_path: Path):
    df = resumen_modelo.sort_values(["modelo", "estrategia"]).copy()
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"\textbf{Modelo} & \textbf{Estrategia} & \textbf{Spearman} & \textbf{nMAE} & \textbf{nRMSE} & \textbf{Entren. (s)} & \textbf{Cob.} \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            " & ".join(
                [
                    latex_escape(row["modelo"]),
                    latex_escape(row["estrategia"]),
                    fmt_float(row["spearman_mean"]),
                    fmt_float(row["nmae_mean"]),
                    fmt_float(row["nrmse_mean"]),
                    fmt_float(row["train_time_s_mean"], digits=3),
                    f"{fmt_float(row['cobertura_pct'], digits=1)}\\%",
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def escribir_tabla_latex_global(resumen_global: pd.DataFrame, out_path: Path):
    df = resumen_global.sort_values("estrategia").copy()
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"\textbf{Estrategia} & \textbf{Spearman} & \textbf{nMAE} & \textbf{nRMSE} & \textbf{Entren. (s)} & \textbf{Cob.} \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            " & ".join(
                [
                    latex_escape(row["estrategia"]),
                    fmt_float(row["spearman_mean"]),
                    fmt_float(row["nmae_mean"]),
                    fmt_float(row["nrmse_mean"]),
                    fmt_float(row["train_time_s_mean"], digits=3),
                    f"{fmt_float(row['cobertura_pct'], digits=1)}\\%",
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    base_dir = Path(args.base_dir)
    if not base_dir.is_dir():
        raise SystemExit(f"No existe base-dir: {base_dir}")

    modelos = [modelo for modelo in args.modelos if modelo not in set(args.exclude_models)]
    if not modelos:
        raise SystemExit("No queda ningun modelo tras aplicar --exclude-models.")

    out_dir = Path(args.out_dir) if args.out_dir else base_dir / "comparacion_estrategias_global"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = cargar_metricas(
        base_dir,
        modelos=modelos,
        protocolos=PROTOCOLOS,
        funciones=args.funciones,
        algoritmos=args.algoritmos,
    )
    if df.empty:
        raise SystemExit(f"No se encontraron metricas en {base_dir}")

    resumen_modelo = resumir(
        df,
        ["modelo", "estrategia"],
        modelos=modelos,
        funciones=args.funciones,
        algoritmos=args.algoritmos,
    )
    resumen_global = resumir(
        df,
        ["estrategia"],
        modelos=modelos,
        funciones=args.funciones,
        algoritmos=args.algoritmos,
    )
    comparacion = comparar_modelos(resumen_modelo)

    df.to_csv(out_dir / "metricas_detalladas_estrategias.csv", index=False)
    resumen_modelo.to_csv(out_dir / "resumen_por_modelo_y_estrategia.csv", index=False)
    resumen_global.to_csv(out_dir / "resumen_global_por_estrategia.csv", index=False)
    comparacion.to_csv(out_dir / "comparacion_acumulativo_vs_no_acumulativo.csv", index=False)

    if not args.no_latex:
        escribir_tabla_latex_modelo(
            resumen_modelo,
            out_dir / "tabla_resumen_por_modelo_y_estrategia.tex",
        )
        escribir_tabla_latex_global(
            resumen_global,
            out_dir / "tabla_resumen_global_por_estrategia.tex",
        )

    print(f"metricas detalladas: {len(df)}")
    print(f"salida: {out_dir}")
    print(resumen_global[["estrategia", "n_metricas", "n_metricas_esperadas", "cobertura_pct", "spearman_mean", "nmae_mean", "nrmse_mean", "train_time_s_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
