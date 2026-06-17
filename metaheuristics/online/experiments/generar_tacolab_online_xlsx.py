"""
Genera un .xlsx por MH (AGE, DE, SHADE) con dos hojas (base / surrogate)
en formato TACOLAB (milestone × función, media del error sobre las semillas).

Estructura de entrada:
  results/cec/online_final_base_{mh}/f{i}/results_{prefix}/results_{i}_10.txt
  results/cec/online_final_surrogate_{mh}/f{i}/results_{prefix}/results_{i}_10.txt

Salida: resultados/tacolab_online/{mh}_tacolab_online.xlsx
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXTRACT = ROOT / "metaheuristics/cec2017/cec2017real/code/extract.py"
OUTDIR  = ROOT / "resultados/tacolab_online"
DIM = 10

VARIANTES = {
    "age":   {
        "base":      "online_final_base_age",
        "surrogate": "online_final_surrogate_age",
    },
    "de":    {
        "base":      "online_final_base_de",
        "surrogate": "online_final_surrogate_de",
    },
    "shade": {
        "base":      "online_final_base_shade",
        "surrogate": "online_final_surrogate_shade",
    },
}


def get_results_subdir(results_dir: Path, fi: int) -> Path:
    """Devuelve el subdirectorio results_* dentro de f{i}/."""
    func_dir = results_dir / f"f{fi}"
    candidates = [d for d in func_dir.iterdir() if d.is_dir() and d.name.startswith("results_")]
    if not candidates:
        raise FileNotFoundError(f"No hay subdirectorio results_* en {func_dir}")
    if len(candidates) > 1:
        raise RuntimeError(f"Múltiples subdirectorios results_* en {func_dir}: {candidates}")
    return candidates[0]


def recopilar_txts(results_dir: Path, tmpdir: Path) -> None:
    for fi in range(1, 31):
        subdir = get_results_subdir(results_dir, fi)
        src = subdir / f"results_{fi}_{DIM}.txt"
        if not src.exists():
            raise FileNotFoundError(f"Falta: {src}")
        shutil.copy(src, tmpdir / src.name)


def ejecutar_extract(tmpdir: Path):
    import pandas as pd
    result = subprocess.run(
        [sys.executable, str(EXTRACT), str(tmpdir)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"extract.py falló:\n{result.stderr}")
    xlsx = tmpdir / f"results_cec2017_{DIM}.xlsx"
    if not xlsx.exists():
        raise FileNotFoundError(f"extract.py no generó {xlsx}")
    return pd.read_excel(xlsx, index_col=0)


def main() -> None:
    import pandas as pd

    OUTDIR.mkdir(parents=True, exist_ok=True)

    for mh, sheets in VARIANTES.items():
        print(f"\n── MH: {mh.upper()}")
        datos = {}
        for label, dirname in sheets.items():
            results_dir = ROOT / "results/cec" / dirname
            print(f"   {label} ({dirname})...", end=" ", flush=True)
            with tempfile.TemporaryDirectory() as tmpdir:
                recopilar_txts(results_dir, Path(tmpdir))
                df = ejecutar_extract(Path(tmpdir))
            cols = ["milestone"] + [f"F{i:02d}" for i in range(1, 31)]
            cols_presentes = [c for c in cols if c in df.columns]
            datos[label] = df[cols_presentes]
            print("ok")

        out = OUTDIR / f"{mh}_tacolab_online.xlsx"
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            for label, df in datos.items():
                df.to_excel(writer, sheet_name=label, index=False)
        print(f"   Guardado: {out}")


if __name__ == "__main__":
    main()
