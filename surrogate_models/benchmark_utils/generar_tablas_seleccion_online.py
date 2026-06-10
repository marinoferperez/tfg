"""
Genera tablas LaTeX para justificar la selección de hiperparámetros online.

Tabla 1 (capítulo): config ganadora por algoritmo — online_seleccion_final.tex
Tabla 2 (capítulo): comparativa config propia vs homogénea — online_comparativa_homogenea_vs_peralgo.tex
Tablas 3-5 (apéndice): top-5 configs por algoritmo — online_apendice_{algo}.tex
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import pandas as pd

PILOT_GLOB = "results/cec/piloto_online_grid_*/f*/runs.csv"
OUT_DIR = Path("memoria/tablas/cap06_ajuste_online")
APENDICE_DIR = OUT_DIR / "apendice"

ALGORITHMS = ["age", "de", "shade"]
ALGO_LABELS = {"age": "AGE", "de": "DE", "shade": "SHADE"}
TOP_N = 5

# Config homogénea minimax
HOMOGENEA = (0.75, 500, 0.25)


def parse_params(adaptacion: str) -> tuple | None:
    m = re.match(r"[a-z]{2}_p(\d+)_c(\d+)_rt(\d+)", adaptacion)
    if not m:
        return None
    return int(m.group(1)) / 100, int(m.group(2)), int(m.group(3)) / 100


def load_pilot() -> pd.DataFrame:
    dfs = [pd.read_csv(f) for f in sorted(glob.glob(PILOT_GLOB))]
    df = pd.concat(dfs, ignore_index=True)
    parsed = df["adaptacion"].map(parse_params)
    df["p"]  = parsed.map(lambda x: x[0] if x else None)
    df["cd"] = parsed.map(lambda x: x[1] if x else None)
    df["rt"] = parsed.map(lambda x: x[2] if x else None)
    return df[df["p"].notna() & (df["cd"] > 0) & (df["p"] > 0)].copy()


def rank_configs(df_algo: pd.DataFrame) -> pd.DataFrame:
    mean_err = (
        df_algo.groupby(["p", "cd", "rt", "cec_funcid"])["cec_error"]
        .mean()
        .reset_index()
    )
    mean_err["rank"] = mean_err.groupby("cec_funcid")["cec_error"].rank(
        ascending=True, method="average"
    )
    rank_medio = mean_err.groupby(["p", "cd", "rt"])["rank"].mean().rename("rank_medio")
    result = rank_medio.reset_index().sort_values("rank_medio").reset_index(drop=True)
    result["pos"] = range(1, len(result) + 1)
    return result


def _bold(s: str) -> str:
    return r"\textbf{" + s + r"}"


def _fmt(v: float, d: int = 3) -> str:
    return f"{v:.{d}f}"


def _p_tex(v: float) -> str:
    return f"${v}$"


def _rt_tex(v: float) -> str:
    return f"${v}$"


# ─── Tabla 1: selección final ────────────────────────────────────────────────

def tabla_seleccion_final(ranked: dict[str, pd.DataFrame]) -> str:
    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        r"    \small",
        r"    \setlength{\tabcolsep}{5pt}",
        r"    \begin{tabular}{lrrrl}",
        r"        \toprule",
        r"        \textbf{Algoritmo} & \textbf{$p$} & \textbf{$cd$} & \textbf{$rt$} "
        r"& \textbf{Rank medio} \\",
        r"        \midrule",
    ]
    for algo in ALGORITHMS:
        best = ranked[algo].iloc[0]
        lines.append(
            f"        {ALGO_LABELS[algo]} & {_p_tex(best.p)} & {int(best.cd)} "
            f"& {_rt_tex(best.rt)} & {_fmt(best.rank_medio)} \\\\"
        )
    lines += [
        r"        \bottomrule",
        r"    \end{tabular}",
        r"    \caption{Configuración del subrogado \textit{online} seleccionada por algoritmo "
        r"según rank medio sobre las 7 funciones de cribado y 10 semillas del piloto.}",
        r"    \label{tab:online_seleccion_final}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# ─── Tabla 2: comparativa homogénea vs per-algo ──────────────────────────────

def tabla_comparativa(ranked: dict[str, pd.DataFrame]) -> str:
    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        r"    \small",
        r"    \setlength{\tabcolsep}{5pt}",
        r"    \begin{tabular}{lrrrr}",
        r"        \toprule",
        r"        \textbf{Algoritmo} & \multicolumn{2}{c}{\textbf{Config propia}} "
        r"& \multicolumn{2}{c}{\textbf{Config homogénea}} \\",
        r"        \cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r"        & \textbf{Rank medio} & \textbf{Pos.} & \textbf{Rank medio} & \textbf{Pos.} \\",
        r"        \midrule",
    ]
    for algo in ALGORITHMS:
        df = ranked[algo]
        best = df.iloc[0]
        hom_row = df[(df["p"] == HOMOGENEA[0]) & (df["cd"] == HOMOGENEA[1]) & (df["rt"] == HOMOGENEA[2])]
        if hom_row.empty:
            hom_rank, hom_pos = "---", "---"
        else:
            hom_rank = _fmt(hom_row.iloc[0].rank_medio)
            hom_pos  = str(int(hom_row.iloc[0].pos))

        best_rank = _bold(_fmt(best.rank_medio))
        best_pos  = _bold("1")

        lines.append(
            f"        {ALGO_LABELS[algo]} & {best_rank} & {best_pos} "
            f"& {hom_rank} & {hom_pos} \\\\"
        )
    lines += [
        r"        \bottomrule",
        r"    \end{tabular}",
        r"    \caption{Comparativa entre la configuración óptima por algoritmo y la "
        r"configuración homogénea ($p$=0.75, $cd$=500, $rt$=0.25) según rank medio "
        r"sobre las 7 funciones de cribado del piloto. Negrita indica el mejor valor por fila.}",
        r"    \label{tab:online_comparativa_homogenea_vs_peralgo}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# ─── Tablas apéndice: top-N por algoritmo ────────────────────────────────────

def tabla_apendice(ranked: pd.DataFrame, algo: str) -> str:
    top = ranked.head(TOP_N)
    label = ALGO_LABELS[algo]
    best_rank = top.iloc[0].rank_medio

    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        r"    \small",
        r"    \setlength{\tabcolsep}{5pt}",
        r"    \begin{tabular}{rrr|rr}",
        r"        \toprule",
        r"        \textbf{$p$} & \textbf{$cd$} & \textbf{$rt$} "
        r"& \textbf{Rank medio} & \textbf{Pos.} \\",
        r"        \midrule",
    ]
    for _, row in top.iterrows():
        rk = _fmt(row.rank_medio)
        if abs(row.rank_medio - best_rank) < 1e-9:
            rk = _bold(rk)
        lines.append(
            f"        {_p_tex(row.p)} & {int(row.cd)} & {_rt_tex(row.rt)} "
            f"& {rk} & {int(row.pos)} \\\\"
        )
    lines += [
        r"        \bottomrule",
        r"    \end{tabular}",
        f"    \\caption{{Top-{TOP_N} configuraciones del subrogado \\textit{{online}} para {label} "
        r"según rank medio sobre las 7 funciones de cribado y 10 semillas del piloto.}",
        f"    \\label{{tab:online_apendice_{algo}}}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    print("Cargando datos del piloto...")
    df = load_pilot()

    ranked = {}
    for algo in ALGORITHMS:
        ranked[algo] = rank_configs(df[df["algoritmo"] == algo])
        print(f"  {ALGO_LABELS[algo]}: mejor config → "
              f"p={ranked[algo].iloc[0].p}, cd={int(ranked[algo].iloc[0].cd)}, "
              f"rt={ranked[algo].iloc[0].rt}, rank={ranked[algo].iloc[0].rank_medio:.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    APENDICE_DIR.mkdir(parents=True, exist_ok=True)

    p = OUT_DIR / "online_seleccion_final.tex"
    p.write_text(tabla_seleccion_final(ranked), encoding="utf-8")
    print(f"  → {p}")

    p = OUT_DIR / "online_comparativa_homogenea_vs_peralgo.tex"
    p.write_text(tabla_comparativa(ranked), encoding="utf-8")
    print(f"  → {p}")

    for algo in ALGORITHMS:
        p = APENDICE_DIR / f"online_apendice_{algo}.tex"
        p.write_text(tabla_apendice(ranked[algo], algo), encoding="utf-8")
        print(f"  → {p}")


if __name__ == "__main__":
    main()
