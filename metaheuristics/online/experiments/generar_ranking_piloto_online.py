"""
Simula el cálculo de ranking de TACOLAB para el piloto online.

Metodología (igual que TACOLAB):
  - Para cada función y milestone: rankear las N configs por error medio (1=mejor)
  - Promediar los ranks sobre las funciones → rank medio por config y milestone
  - Mostrar hito 100% como principal; incluir hitos intermedios en apéndice

Salida: resultados/tacolab_piloto_online/ranking_{age,de,shade}_piloto.tex
        resultados/tacolab_piloto_online/ranking_{age,de,shade}_piloto_milestones.tex
"""
from __future__ import annotations

from pathlib import Path
import math
import pandas as pd
import numpy as np

ROOT   = Path(__file__).resolve().parents[3]
INDIR  = ROOT / "resultados/tacolab_piloto_online"
OUTDIR = ROOT / "resultados/tacolab_piloto_online"

TOL_RANK    = 1e-12
FUNCS       = ["F03", "F05", "F09", "F15", "F19", "F24", "F27"]
MHS         = ["age", "de", "shade"]
MS_ALL      = [1, 2, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
MS_SHOW     = [10, 20, 50, 100]   # hitos a mostrar en tabla resumen
MS_MAIN     = 100                  # hito principal


def parse_label(label: str) -> tuple[float, int, float]:
    """Convierte 'p75_cd500_rt25' → (0.75, 500, 0.25)."""
    parts = label.split("_")
    p  = int(parts[0][1:]) / 100
    cd = int(parts[1][2:])
    rt = int(parts[2][2:]) / 100
    return p, cd, rt


def load_data(mh: str) -> dict[str, pd.DataFrame]:
    """Carga todas las hojas del xlsx. Devuelve {label: df(milestone × funcs)}."""
    xl = pd.ExcelFile(INDIR / f"{mh}_tacolab_piloto.xlsx")
    return {s: xl.parse(s) for s in xl.sheet_names}


def compute_rankings(data: dict[str, pd.DataFrame],
                     milestones: list[int]) -> pd.DataFrame:
    """
    Devuelve DataFrame (config × milestone) con rank medio.
    Para cada milestone y función: rankea configs por error (1=mejor).
    Promedia sobre funciones.
    """
    configs = [cfg for cfg in data if parse_label(cfg)[1] != 0]
    results = {ms: {} for ms in milestones}

    for ms in milestones:
        # Construir matrix: config × función con errores al hito ms.
        # TACOLAB rankea por valor medio y trata como empate diferencias
        # numéricas despreciables, por ejemplo errores del orden de 1e-14.
        err = {}
        for cfg in configs:
            df = data[cfg]
            row = df[df["milestone"] == ms]
            if row.empty:
                err[cfg] = {f: np.nan for f in FUNCS}
            else:
                err[cfg] = {f: float(row.iloc[0][f])
                            for f in FUNCS if f in row.columns}

        err_df = pd.DataFrame(err, index=FUNCS).T  # config × func

        ranks = rankear_dataframe(err_df)
        mean_ranks = ranks.mean(axis=1)

        for cfg in configs:
            results[ms][cfg] = mean_ranks[cfg]

    return pd.DataFrame(results, index=configs)   # config × milestone


def compute_victorias(data: dict[str, pd.DataFrame], ms: int = 100) -> pd.Series:
    """Nº de funciones en que cada config obtiene rank=1 al hito ms."""
    configs = [cfg for cfg in data if parse_label(cfg)[1] != 0]
    wins = {cfg: 0 for cfg in configs}

    for func in FUNCS:
        errs = {}
        for cfg in configs:
            df = data[cfg]
            row = df[df["milestone"] == ms]
            if not row.empty and func in row.columns:
                errs[cfg] = float(row.iloc[0][func])
        if not errs:
            continue
        min_val = min(normalizar_cero(v) for v in errs.values())
        for cfg, v in errs.items():
            if math.isclose(normalizar_cero(v), min_val, rel_tol=TOL_RANK, abs_tol=TOL_RANK):
                wins[cfg] += 1

    return pd.Series(wins)


def normalizar_cero(valor: float) -> float:
    """Evita que ruido numérico cercano a cero rompa empates prácticos."""
    return 0.0 if abs(float(valor)) <= TOL_RANK else float(valor)


def rankear_dataframe(err_df: pd.DataFrame) -> pd.DataFrame:
    """Rankea configuraciones por función usando empates con tolerancia."""
    ranks = pd.DataFrame(index=err_df.index, columns=err_df.columns, dtype=float)
    for func in err_df.columns:
        valores = {
            cfg: normalizar_cero(err_df.loc[cfg, func])
            for cfg in err_df.index
            if not pd.isna(err_df.loc[cfg, func])
        }
        ordenados = sorted(valores.items(), key=lambda item: (item[1], item[0]))
        posicion = 1
        i = 0
        while i < len(ordenados):
            j = i + 1
            while j < len(ordenados) and math.isclose(
                ordenados[i][1],
                ordenados[j][1],
                rel_tol=TOL_RANK,
                abs_tol=TOL_RANK,
            ):
                j += 1
            rank = (posicion + posicion + (j - i) - 1) / 2.0
            for k in range(i, j):
                ranks.loc[ordenados[k][0], func] = rank
            posicion += j - i
            i = j
    return ranks


def gen_tex_ranking(rank_df: pd.DataFrame, wins: pd.Series) -> str:
    """
    Tabla principal: configuraciones ordenadas por ranking medio al hito final.
    Ordenada por rank al hito 100%. Bold en el mejor de cada columna.
    """
    order   = rank_df[MS_MAIN].sort_values().index.tolist()
    best_rank = rank_df[MS_MAIN].min()
    best_wins   = wins.max()

    col_spec = "lllrr"
    header   = (
        r"\textbf{$p$} & \textbf{$cd$} & \textbf{$rt$} & "
        r"\textbf{Rank medio} & \textbf{Vict.} \\"
    )

    lines = [
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for cfg in order:
        p, cd, rt = parse_label(cfg)
        v = rank_df.loc[cfg, MS_MAIN]
        rank_str = f"{v:.3f}"
        if abs(v - best_rank) < 1e-9:
            rank_str = r"\textbf{" + rank_str + "}"
        w = wins[cfg]
        w_str = r"\textbf{" + str(w) + "}" if w == best_wins else str(w)
        lines.append(f"{p} & {cd} & {rt} & {rank_str} & {w_str} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    for mh in MHS:
        mh_upper = mh.upper()
        print(f"\n── {mh_upper}")
        data = load_data(mh)

        rank_df = compute_rankings(data, MS_ALL)
        wins    = compute_victorias(data, MS_MAIN)

        tex = gen_tex_ranking(rank_df, wins)
        p = OUTDIR / f"ranking_{mh}_piloto.tex"
        p.write_text(tex, encoding="utf-8")
        print(f"   → {p}")

        # Mostrar top-5 en consola
        top5 = rank_df[MS_MAIN].sort_values().head(5)
        print(f"   Top-5 al 100%:")
        for cfg, r in top5.items():
            p2, cd, rt = parse_label(cfg)
            print(f"      p={p2} cd={cd} rt={rt}  rank={r:.3f}  wins={wins[cfg]}")


if __name__ == "__main__":
    main()
