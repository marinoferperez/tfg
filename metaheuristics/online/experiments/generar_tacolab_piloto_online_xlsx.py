"""
Genera un .xlsx por MH (AGE, DE, SHADE) con una hoja por configuración del piloto
online en formato TACOLAB (milestone × función, media del error sobre 10 semillas).

Fuente: results/cec/piloto_online_grid_*/
Salida: resultados/tacolab_piloto_online/{age,de,shade}_tacolab_piloto.xlsx
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[3]
EXTRACT = ROOT / "metaheuristics/cec2017/cec2017real/code/extract.py"
OUTDIR  = ROOT / "resultados/tacolab_piloto_online"
DIM     = 10

PREFIX  = {"age": "ao", "de": "do", "shade": "so"}
FUNCS   = [3, 5, 9, 15, 19, 24, 27]   # funciones de cribado


def get_configs() -> list[tuple[str, Path]]:
    """Devuelve lista de (label, path) ordenada por label."""
    pattern = re.compile(r"piloto_online_grid_(p\d+_cd\d+_rt\d+)$")
    configs = []
    for d in sorted((ROOT / "results/cec").iterdir()):
        m = pattern.match(d.name)
        if m:
            configs.append((m.group(1), d))
    return configs


def get_results_subdir(func_dir: Path, mh: str) -> Path:
    prefix = PREFIX[mh]
    candidates = [d for d in func_dir.iterdir()
                  if d.is_dir() and d.name.startswith(f"results_{prefix}_")]
    if not candidates:
        raise FileNotFoundError(f"No hay results_{prefix}_* en {func_dir}")
    return candidates[0]


def leer_milestone_tabla(config_dir: Path, mh: str) -> "pd.DataFrame":
    """Lee los txt de las funciones de cribado y devuelve milestone × función (media)."""
    import pandas as pd

    dfs = []
    for fi in FUNCS:
        func_dir = config_dir / f"f{fi}"
        subdir   = get_results_subdir(func_dir, mh)
        src      = subdir / f"results_{fi}_{DIM}.txt"
        if not src.exists():
            raise FileNotFoundError(f"Falta: {src}")
        dfs.append(pd.read_csv(src))

    df = pd.concat(dfs, ignore_index=True)
    # Media del error por función y milestone (igual que extract.py)
    mean_df = (
        df.groupby(["funcid", "milestone"])["error"]
        .mean()
        .reset_index()
    )
    pivot = mean_df.pivot(index="milestone", columns="funcid", values="error")
    pivot.columns = [f"F{int(c):02d}" for c in pivot.columns]
    pivot = pivot.reset_index().rename(columns={"milestone": "milestone"})
    return pivot


def main() -> None:
    import pandas as pd

    OUTDIR.mkdir(parents=True, exist_ok=True)
    configs = get_configs()
    print(f"Configuraciones encontradas: {len(configs)}")

    func_cols = [f"F{fi:02d}" for fi in FUNCS]

    for mh in ["age", "de", "shade"]:
        print(f"\n── MH: {mh.upper()}")
        sheets = {}
        for label, config_dir in configs:
            print(f"   {label}...", end=" ", flush=True)
            df = leer_milestone_tabla(config_dir, mh)
            cols = ["milestone"] + [c for c in func_cols if c in df.columns]
            sheets[label] = df[cols]
            print("ok")

        out = OUTDIR / f"{mh}_tacolab_piloto.xlsx"
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            for label, df in sheets.items():
                df.to_excel(writer, sheet_name=label, index=False)
        print(f"   Guardado: {out}")


if __name__ == "__main__":
    main()
