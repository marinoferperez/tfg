#!/usr/bin/env python3
"""
Genera las tablas offline en el nuevo formato (ranking por algoritmo):

  - comparativa_estrategias_offline_v2.tex   (Tabla 7.6 rediseñada)
  - comparativa_modelos_offline_v2.tex        (Tabla 7.7 rediseñada)

Criterio principal: correlación de Spearman (mayor = mejor).
Tiempos de entrenamiento y predicción: agregados entre algoritmos.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(
    "results/cec/cec2017_d10_tam50_reinicio_seleccionado"
    "/benchmarking_surrogates_offline_next/future_next"
)
OUT_DIR = Path("memoria/tablas")

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
ESTRATEGIAS = ["no_acumulativo", "acumulativo"]
BLOQUES_NO_ACUM = ["1-20", "21-40", "41-60", "61-80"]
BLOQUES_ACUM = ["1-20", "1-40", "1-60", "1-80"]
BLOQUES_POR_ESTRATEGIA = {
    "no_acumulativo": BLOQUES_NO_ACUM,
    "acumulativo": BLOQUES_ACUM,
}


def cargar_datos() -> pd.DataFrame:
    """Lee todos los JSONs por bloque y devuelve un DataFrame detallado."""
    rows = []
    for modelo in MODELOS:
        for estrategia in ESTRATEGIAS:
            bloques = BLOQUES_POR_ESTRATEGIA[estrategia]
            base_est = BASE_DIR / modelo / estrategia
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
                    for bloque in bloques:
                        json_path = alg_dir / bloque / f"{modelo}_metricas.json"
                        if not json_path.exists():
                            continue
                        data = json.loads(json_path.read_text(encoding="utf-8"))
                        rows.append(
                            {
                                "modelo": modelo,
                                "estrategia": estrategia,
                                "funcion": funcion,
                                "algoritmo": algoritmo,
                                "bloque": bloque,
                                "spearman": float(data.get("spearman") or 0),
                                "nrmse": float(data.get("nrmse") or 0),
                                "nmae": float(data.get("nmae") or 0),
                                "train_time_s": float(data.get("train_time_s") or 0),
                                "predict_time_s": float(
                                    data.get("predict_time_s") or 0
                                ),
                            }
                        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tabla 7.6 — estrategia acumulativa vs no acumulativa, por algoritmo
# ---------------------------------------------------------------------------

def generar_tabla_estrategias(df: pd.DataFrame) -> str:
    """
    Por cada algoritmo: victorias de no_acumulativo (%), Spearman de cada
    estrategia, y tiempos de entrenamiento de cada estrategia.

    Las victorias se cuentan por bloque de validación comparable (excluye el
    primer bloque 1-20, donde ambas estrategias usan los mismos datos de
    entrenamiento). Los bloques se emparejan por ventana de validación:
      no_acum "21-40" ↔ acum "1-40"  (ambos validan en 40-60%)
      no_acum "41-60" ↔ acum "1-60"  (ambos validan en 60-80%)
      no_acum "61-80" ↔ acum "1-80"  (ambos validan en 80-100%)
    """
    # Mapeo: bloque_no_acum → bloque_acum (misma ventana de validación)
    PARES_BLOQUES = [
        ("21-40", "1-40"),
        ("41-60", "1-60"),
        ("61-80", "1-80"),
    ]

    # Construir DataFrame de comparaciones emparejadas por bloque de validación
    rows_cmp = []
    for bloque_no, bloque_ac in PARES_BLOQUES:
        df_no = df[(df["estrategia"] == "no_acumulativo") & (df["bloque"] == bloque_no)]
        df_ac = df[(df["estrategia"] == "acumulativo") & (df["bloque"] == bloque_ac)]
        merged = df_no.merge(
            df_ac,
            on=["funcion", "modelo", "algoritmo"],
            suffixes=("_no", "_ac"),
        )
        merged["no_acum_gana"] = merged["spearman_no"] > merged["spearman_ac"]
        merged["bloque_validacion"] = bloque_no  # etiqueta del par
        rows_cmp.append(merged)

    df_cmp = pd.concat(rows_cmp, ignore_index=True)

    # Victorias por algoritmo (sobre funciones × modelos × 3 bloques comparables)
    victorias = df_cmp.groupby("algoritmo")["no_acum_gana"].agg(["sum", "count"])

    # Spearman medio por algoritmo (solo bloques comparables, excluye 1-20)
    df_no_cmp = df[
        (df["estrategia"] == "no_acumulativo") & (df["bloque"] != "1-20")
    ]
    df_ac_cmp = df[
        (df["estrategia"] == "acumulativo") & (df["bloque"] != "1-20")
    ]
    sp_no = df_no_cmp.groupby("algoritmo")["spearman"].mean()
    sp_ac = df_ac_cmp.groupby("algoritmo")["spearman"].mean()

    # Tiempos medios por algoritmo (sobre modelos × funciones × bloques comparables)
    t_no = df_no_cmp.groupby("algoritmo")["train_time_s"].mean()
    t_ac = df_ac_cmp.groupby("algoritmo")["train_time_s"].mean()

    lines = [
        r"\begin{tabular}{l r r r r r}",
        r"\toprule",
        (
            r"\textbf{Algoritmo} & \textbf{Vict. no acum.} "
            r"& \textbf{Spearman no acum.} & \textbf{Spearman acum.} "
            r"& \textbf{Entren. no acum. (s)} & \textbf{Entren. acum. (s)} \\"
        ),
        r"\midrule",
    ]

    for algo in ALGORITMOS:
        vic_sum = int(victorias.loc[algo, "sum"]) if algo in victorias.index else 0
        vic_total = int(victorias.loc[algo, "count"]) if algo in victorias.index else 0
        sno = sp_no.get(algo, float("nan"))
        sac = sp_ac.get(algo, float("nan"))
        tno = t_no.get(algo, float("nan"))
        tac = t_ac.get(algo, float("nan"))

        sno_fmt = r"\textbf{" + f"{sno:.4f}" + r"}" if sno > sac else f"{sno:.4f}"
        sac_fmt = r"\textbf{" + f"{sac:.4f}" + r"}" if sac >= sno else f"{sac:.4f}"

        lines.append(
            f"{algo.upper()} & {vic_sum}/{vic_total} & {sno_fmt} "
            f"& {sac_fmt} & {tno:.4f} & {tac:.4f} \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tabla 7.7 — competición de modelos, ranking por algoritmo
# ---------------------------------------------------------------------------

def generar_tabla_modelos(df: pd.DataFrame) -> str:
    """
    Para la estrategia no_acumulativa:
    - Rank AGE, Rank DE, Rank SHADE: rango medio (1=mejor Spearman) sobre
      funciones × bloques para cada algoritmo.
    - Rank medio: promedio de los tres rankings anteriores.
    - Entren. (s) y Pred. (s): media agregada sobre algoritmos, funciones y bloques.
    """
    df_no = df[df["estrategia"] == "no_acumulativo"].copy()

    # Redondeo a 4 decimales antes del ranking (criterio TACOLAB)
    df_no["spearman"] = df_no["spearman"].round(4)

    # Ranking por (funcion, bloque, algoritmo): rank entre modelos
    df_no["rank"] = df_no.groupby(["funcion", "bloque", "algoritmo"])[
        "spearman"
    ].rank(ascending=False, method="min")

    # Rank medio por (modelo, algoritmo)
    rank_medio = (
        df_no.groupby(["modelo", "algoritmo"])["rank"]
        .mean()
        .unstack("algoritmo")
    )

    # Rank global (promedio de los 3 algoritmos)
    rank_medio["rank_global"] = rank_medio[ALGORITMOS].mean(axis=1)

    # Tiempos agregados (sobre algoritmos, funciones, bloques)
    tiempos = df_no.groupby("modelo")[["train_time_s", "predict_time_s"]].mean()

    # Ordenar por rank global
    rank_medio = rank_medio.join(tiempos).sort_values("rank_global")

    # Determinar mejor rank por columna para poner negrita
    best_rank_age = rank_medio["age"].min()
    best_rank_de = rank_medio["de"].min()
    best_rank_shade = rank_medio["shade"].min()
    best_rank_global = rank_medio["rank_global"].min()
    best_train = tiempos["train_time_s"].min()
    best_pred = tiempos["predict_time_s"].min()

    def fmt_rank(val, best):
        s = f"{val:.4f}"
        return r"\textbf{" + s + r"}" if abs(val - best) < 1e-9 else s

    def fmt_time(val, best):
        s = f"{val:.4f}"
        return r"\textbf{" + s + r"}" if abs(val - best) < 1e-9 else s

    lines = [
        r"\begin{tabular}{l r r r r r r}",
        r"\toprule",
        (
            r"\textbf{Modelo} & \textbf{Rank AGE} & \textbf{Rank DE} "
            r"& \textbf{Rank SHADE} & \textbf{Rank medio} "
            r"& \textbf{Entren. (s)} & \textbf{Pred. (s)} \\"
        ),
        r"\midrule",
    ]

    for modelo, row in rank_medio.iterrows():
        name = MODELOS_DISPLAY.get(modelo, modelo.upper())
        lines.append(
            f"{name} & "
            f"{fmt_rank(row['age'], best_rank_age)} & "
            f"{fmt_rank(row['de'], best_rank_de)} & "
            f"{fmt_rank(row['shade'], best_rank_shade)} & "
            f"{fmt_rank(row['rank_global'], best_rank_global)} & "
            f"{fmt_time(row['train_time_s'], best_train)} & "
            f"{fmt_time(row['predict_time_s'], best_pred)} \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def wrap_table(tabular: str, caption: str, label: str) -> str:
    return (
        "\\begin{table}[H]\n"
        "    \\centering\n"
        "    \\small\n"
        "    \\begin{adjustbox}{max width=\\textwidth}\n"
        + tabular
        + "\n    \\end{adjustbox}\n"
        f"    \\caption{{{caption}}}\n"
        f"    \\label{{{label}}}\n"
        "\\end{table}"
    )


def main():
    print("Cargando datos...")
    df = cargar_datos()
    print(f"  {len(df)} filas cargadas.")
    print(f"  Modelos:     {sorted(df['modelo'].unique())}")
    print(f"  Algoritmos:  {sorted(df['algoritmo'].unique())}")
    print(f"  Funciones:   {sorted(df['funcion'].unique())}")
    print(f"  Estrategias: {sorted(df['estrategia'].unique())}")
    print(f"  Bloques:     {sorted(df['bloque'].unique())}")

    # Tabla 7.6
    print("\nGenerando tabla de estrategias...")
    tab_estrategias = generar_tabla_estrategias(df)
    out_est = OUT_DIR / "comparativa_estrategias_offline_v2.tex"
    out_est.write_text(
        wrap_table(
            tab_estrategias,
            caption=(
                "Comparativa de estrategias \\textit{offline} por algoritmo. "
                "Victorias: número de combinaciones (función~$\\times$~bloque~$\\times$~modelo) "
                "sobre los bloques comparables posteriores al primero "
                "en las que la estrategia no acumulativa supera a la acumulativa en Spearman."
            ),
            label="tab:comparativa_estrategias_offline",
        ),
        encoding="utf-8",
    )
    print(f"  Escrito: {out_est}")

    # Tabla 7.7
    print("\nGenerando tabla de modelos con ranking...")
    tab_modelos = generar_tabla_modelos(df)
    out_mod = OUT_DIR / "comparativa_modelos_offline_v2.tex"
    out_mod.write_text(
        wrap_table(
            tab_modelos,
            caption=(
                "Competición de modelos bajo la estrategia no acumulativa. "
                "Rank~AGE/DE/SHADE: posición media (1\\,=\\,mejor Spearman) sobre "
                "funciones~$\\times$~bloques de validación para cada algoritmo. "
                "Tiempos agregados sobre algoritmos, funciones y bloques."
            ),
            label="tab:comparativa_modelos_offline_v2",
        ),
        encoding="utf-8",
    )
    print(f"  Escrito: {out_mod}")

    # Resumen numérico en pantalla
    print("\n--- Ranking medio de modelos (no_acumulativo) ---")
    df_no = df[df["estrategia"] == "no_acumulativo"].copy()
    df_no["rank"] = df_no.groupby(["funcion", "bloque", "algoritmo"])[
        "spearman"
    ].rank(ascending=False, method="min")
    resumen = (
        df_no.groupby(["modelo", "algoritmo"])["rank"]
        .mean()
        .unstack("algoritmo")
    )
    resumen["rank_global"] = resumen[ALGORITMOS].mean(axis=1)
    tiempos = df_no.groupby("modelo")[["train_time_s", "predict_time_s"]].mean()
    resumen = resumen.join(tiempos).sort_values("rank_global")
    print(resumen.to_string())

    print("\n--- Victorias no_acumulativo por algoritmo (bloques comparables) ---")
    PARES = [("21-40", "1-40"), ("41-60", "1-60"), ("61-80", "1-80")]
    rows_cmp = []
    for bloque_no, bloque_ac in PARES:
        df_no = df[(df["estrategia"] == "no_acumulativo") & (df["bloque"] == bloque_no)]
        df_ac = df[(df["estrategia"] == "acumulativo") & (df["bloque"] == bloque_ac)]
        merged = df_no.merge(df_ac, on=["funcion", "modelo", "algoritmo"], suffixes=("_no", "_ac"))
        merged["no_acum_gana"] = merged["spearman_no"] > merged["spearman_ac"]
        rows_cmp.append(merged)
    df_cmp2 = pd.concat(rows_cmp, ignore_index=True)
    print(
        df_cmp2.groupby("algoritmo")["no_acum_gana"]
        .agg(pct_victorias=lambda x: x.mean() * 100, victorias="sum", total="count")
        .to_string()
    )


if __name__ == "__main__":
    main()
