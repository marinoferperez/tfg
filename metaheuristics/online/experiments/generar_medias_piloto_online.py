from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
INDIR = ROOT / "resultados/tacolab_piloto_online"
OUTDIR = ROOT / "resultados/tacolab_piloto_online"

FUNCS = ["F03", "F05", "F09", "F15", "F19", "F24", "F27"]
MHS = ["age", "de", "shade"]
MS_MAIN = 100
TOL_RANK = 1e-12


def parse_label(label: str) -> tuple[float, int, float]:
    parts = label.split("_")
    p = int(parts[0][1:]) / 100
    cd = int(parts[1][2:])
    rt = int(parts[2][2:]) / 100
    return p, cd, rt


def normalizar_cero(valor: float) -> float:
    return 0.0 if abs(float(valor)) <= TOL_RANK else float(valor)


def cargar_data(mh: str) -> dict[str, pd.DataFrame]:
    xl = pd.ExcelFile(INDIR / f"{mh}_tacolab_piloto.xlsx")
    return {
        sheet: xl.parse(sheet)
        for sheet in xl.sheet_names
        if parse_label(sheet)[1] != 0
    }


def valores_finales(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for cfg, df in data.items():
        row = df[df["milestone"] == MS_MAIN]
        if row.empty:
            raise RuntimeError(f"No existe milestone {MS_MAIN} para {cfg}")
        item = {"config": cfg}
        for func in FUNCS:
            item[func] = float(row.iloc[0][func])
        rows.append(item)
    return pd.DataFrame(rows).set_index("config")


def contar_victorias(values: pd.DataFrame) -> pd.Series:
    wins = pd.Series(0, index=values.index, dtype=int)
    for func in FUNCS:
        col = values[func].map(normalizar_cero)
        min_val = col.min()
        winners = col[(col - min_val).abs() <= TOL_RANK].index
        for cfg in winners:
            wins[cfg] += 1
    return wins


def fmt_sci(value: float) -> str:
    return f"{float(value):.4e}"


def tex_escape_config(cfg: str) -> str:
    p, cd, rt = parse_label(cfg)
    return rf"$p={p:g}, cd={cd}, rt={rt:g}$"


def generar_tex(values: pd.DataFrame, wins: pd.Series) -> str:
    ordered = wins.sort_values(ascending=False).index.tolist()
    ordered = sorted(ordered, key=lambda cfg: (-wins[cfg], cfg))
    col_spec = "l" + "r" * len(ordered)
    header = "{} & " + " & ".join(tex_escape_config(cfg) for cfg in ordered) + r" \\"

    lines = [
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for func in FUNCS:
        row_values = values.loc[ordered, func].map(normalizar_cero)
        min_val = row_values.min()
        cells = []
        for cfg in ordered:
            text = fmt_sci(values.loc[cfg, func])
            if abs(normalizar_cero(values.loc[cfg, func]) - min_val) <= TOL_RANK:
                text = rf"\textbf{{{text}}}"
            cells.append(text)
        lines.append(f"{func} & " + " & ".join(cells) + r" \\")

    lines.append(r"\midrule")
    lines.append(
        "Best & "
        + " & ".join(
            rf"\textbf{{{int(wins[cfg])}}}" if wins[cfg] == wins.max() else str(int(wins[cfg]))
            for cfg in ordered
        )
        + r" \\"
    )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def generar_csv(values: pd.DataFrame, wins: pd.Series, path: Path) -> None:
    ordered = sorted(wins.index, key=lambda cfg: (-wins[cfg], cfg))
    out = values.loc[ordered].T
    out.loc["Best"] = wins.loc[ordered]
    out.to_csv(path)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for mh in MHS:
        data = cargar_data(mh)
        values = valores_finales(data)
        wins = contar_victorias(values)
        tex = generar_tex(values, wins)
        tex_path = OUTDIR / f"media_{mh}_piloto.tex"
        csv_path = OUTDIR / f"media_{mh}_piloto.csv"
        tex_path.write_text(tex, encoding="utf-8")
        generar_csv(values, wins, csv_path)
        print(f"{mh.upper()}: {tex_path}")


if __name__ == "__main__":
    main()
