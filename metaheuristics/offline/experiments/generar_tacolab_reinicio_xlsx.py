"""
Genera un .xlsx por algoritmo (AGE, DE, SHADE) con una hoja por variante de reinicio
en formato TACOLAB (milestone × función, media del error sobre 51 semillas).

Variantes:
  - base        → cec2017_d10_tam50              (sin reinicio)
  - reinicio-1  → cec2017_d10_tam50_reinicio_pat001
  - reinicio-3  → cec2017_d10_tam50_reinicio_pat003
  - reinicio-5  → cec2017_d10_tam50_reinicio_pat005
  - reinicio-7  → cec2017_d10_tam50_reinicio_pat007
  - reinicio-10 → cec2017_d10_tam50_reinicio_pat01

Salida: resultados/tacolab_reinicio_{age,de,shade}.xlsx
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

VARIANTES: list[tuple[str, str]] = [
    ("base",        "cec2017_d10_tam50"),
    ("reinicio-1",  "cec2017_d10_tam50_reinicio_pat001"),
    ("reinicio-3",  "cec2017_d10_tam50_reinicio_pat003"),
    ("reinicio-5",  "cec2017_d10_tam50_reinicio_pat005"),
    ("reinicio-7",  "cec2017_d10_tam50_reinicio_pat007"),
    ("reinicio-10", "cec2017_d10_tam50_reinicio_pat01"),
]

ALGORITMOS = ["age", "de", "shade"]
DIM = 10
EXTRACT = ROOT / "metaheuristics/cec2017/cec2017real/code/extract.py"
OUTDIR   = ROOT / "resultados/tacolab_reinicio"


def recopilar_txts(results_dir: Path, algo: str, tmpdir: Path) -> None:
    """Copia los 30 txt de results_{algo}/ al tmpdir."""
    for fi in range(1, 31):
        src = results_dir / f"f{fi}" / f"results_{algo}" / f"results_{fi}_{DIM}.txt"
        if not src.exists():
            raise FileNotFoundError(f"Falta: {src}")
        shutil.copy(src, tmpdir / src.name)


def ejecutar_extract(tmpdir: Path) -> pd.DataFrame:
    """Ejecuta extract.py en tmpdir y devuelve el DataFrame resultante."""
    result = subprocess.run(
        [sys.executable, str(EXTRACT), str(tmpdir)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"extract.py falló:\n{result.stderr}")
    xlsx = tmpdir / f"results_cec2017_{DIM}.xlsx"
    if not xlsx.exists():
        raise FileNotFoundError(f"extract.py no generó {xlsx}")
    df = pd.read_excel(xlsx, index_col=0)
    return df


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # datos[algo][variante_label] = DataFrame
    datos: dict[str, dict[str, pd.DataFrame]] = {a: {} for a in ALGORITMOS}

    for label, dirname in VARIANTES:
        results_dir = ROOT / "results/cec" / dirname
        print(f"\n── Variante: {label} ({dirname})")
        for algo in ALGORITMOS:
            print(f"   {algo}...", end=" ", flush=True)
            with tempfile.TemporaryDirectory() as tmpdir:
                recopilar_txts(results_dir, algo, Path(tmpdir))
                df = ejecutar_extract(Path(tmpdir))
            datos[algo][label] = df
            print("ok")

    for algo in ALGORITMOS:
        out_xlsx = OUTDIR / f"tacolab_reinicio_{algo}.xlsx"
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
            for label, _ in VARIANTES:
                df = datos[algo][label]
                # Mantenemos solo milestone y las 30 funciones
                cols = ["milestone"] + [f"F{i:02d}" for i in range(1, 31)]
                cols_presentes = [c for c in cols if c in df.columns]
                df[cols_presentes].to_excel(writer, sheet_name=label, index=False)
        print(f"Guardado: {out_xlsx}")


if __name__ == "__main__":
    main()
