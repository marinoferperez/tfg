"""
Convierte las tablas de media media_{age,de,shade}_online.tex en un único
.xlsx con una hoja por metaheurística para enviar a Daniel.
"""
from pathlib import Path
import re
import pandas as pd

INDIR = Path(__file__).parent
OUT   = INDIR / "media_online_para_daniel.xlsx"

MHS = ["age", "de", "shade"]


def parse_media_tex(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8")
    rows = []
    for line in text.splitlines():
        line = line.strip().rstrip("\\").strip()
        if not re.match(r"F\d+|Best", line):
            continue
        parts = [c.strip() for c in line.split("&")]
        if len(parts) < 3:
            continue
        func = parts[0]
        base = parts[1]
        surr = parts[2]
        rows.append({"Función": func, "Base": base, "Surrogate": surr})
    return pd.DataFrame(rows)


with pd.ExcelWriter(OUT, engine="openpyxl") as writer:
    for mh in MHS:
        tex = INDIR / f"media_{mh}_online.tex"
        df = parse_media_tex(tex)
        df.to_excel(writer, sheet_name=mh.upper(), index=False)
        print(f"Hoja {mh.upper()}: {len(df)} filas")

print(f"\nGuardado: {OUT}")
