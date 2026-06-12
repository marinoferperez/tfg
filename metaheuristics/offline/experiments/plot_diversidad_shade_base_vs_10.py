"""
Genera la figura comparando diversidad de SHADE base vs SHADE-10 sobre una función CEC2017.
Usa la semilla representativa (mediana de diversidad final) de entre las disponibles.

Uso:
    python metaheuristics/offline/experiments/plot_diversidad_shade_base_vs_10.py --funcid 10
    python metaheuristics/offline/experiments/plot_diversidad_shade_base_vs_10.py --funcid 12 --outdir memoria/figuras/reinicio
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

DIR_BASE = ROOT / "results/cec/cec2017_d10_tam50"
DIR_SEL  = ROOT / "results/cec/cec2017_d10_tam50_reinicio_seleccionado"

MAX_EVALS = 100_000
COL_DIV   = "div_dist_euclidea_normalizada"
COL_EVAL  = "evaluaciones"


def _metricas_shade(base_dir: Path, funcid: int, semilla: int) -> Path:
    d = base_dir / f"f{funcid}" / "metricas_runs" / "cec2017" / "shade"
    candidatos = list(d.glob(f"shade_cec2017_f{funcid}_d10_s{semilla}*"))
    if not candidatos:
        raise FileNotFoundError(f"No se encontraron métricas en {d} para semilla {semilla}")
    return candidatos[0]


def _reinicios_shade(base_dir: Path, funcid: int, semilla: int) -> pd.DataFrame | None:
    try:
        ruta = _metricas_shade(base_dir, funcid, semilla) / "reinicios_elitistas.csv"
        if ruta.exists():
            return pd.read_csv(ruta)
    except FileNotFoundError:
        pass
    return None


def _semillas_disponibles(base_dir: Path, funcid: int) -> list[int]:
    d = base_dir / f"f{funcid}" / "metricas_runs" / "cec2017" / "shade"
    if not d.exists():
        return []
    semillas = []
    for p in d.iterdir():
        if p.is_dir():
            partes = p.name.split("_s")
            if len(partes) >= 2:
                try:
                    semillas.append(int(partes[-1].split("_")[0].split("r")[0]))
                except ValueError:
                    pass
    return sorted(set(semillas))


def _semilla_representativa(base_dir: Path, funcid: int, semillas: list[int]) -> int:
    divs_finales = []
    for s in semillas:
        try:
            ruta = _metricas_shade(base_dir, funcid, s) / "resultados_shade_cec2017_f10_d10.csv"
            csv_name = f"resultados_shade_cec2017_f{funcid}_d10.csv"
            ruta = _metricas_shade(base_dir, funcid, s) / csv_name
            df = pd.read_csv(ruta)
            divs_finales.append(df[COL_DIV].iloc[-1])
        except Exception:
            divs_finales.append(np.nan)
    arr = np.array(divs_finales, dtype=float)
    mediana = np.nanmedian(arr)
    idx = int(np.nanargmin(np.abs(arr - mediana)))
    return semillas[idx]


def _cargar_diversidad(base_dir: Path, funcid: int, semilla: int) -> pd.DataFrame:
    csv_name = f"resultados_shade_cec2017_f{funcid}_d10.csv"
    ruta = _metricas_shade(base_dir, funcid, semilla) / csv_name
    df = pd.read_csv(ruta)[[COL_EVAL, COL_DIV]]
    return df[df[COL_EVAL] <= MAX_EVALS]


def generar_figura(funcid: int, outdir: Path) -> None:
    semillas_base = _semillas_disponibles(DIR_BASE, funcid)
    semillas_sel  = _semillas_disponibles(DIR_SEL,  funcid)
    semillas_comunes = sorted(set(semillas_base) & set(semillas_sel))
    if not semillas_comunes:
        raise RuntimeError(f"No hay semillas comunes para f{funcid} entre base y seleccionado")

    semilla = _semilla_representativa(DIR_BASE, funcid, semillas_comunes)
    print(f"Semilla representativa para f{funcid}: {semilla}")

    df_base = _cargar_diversidad(DIR_BASE, funcid, semilla)
    df_sel  = _cargar_diversidad(DIR_SEL,  funcid, semilla)
    reinicios = _reinicios_shade(DIR_SEL, funcid, semilla)

    fig, ax = plt.subplots(figsize=(7, 3.5))

    ax.plot(df_base[COL_EVAL], df_base[COL_DIV],
            color="tab:green", linewidth=1.2, label="SHADE (sin reinicio)")
    ax.plot(df_sel[COL_EVAL], df_sel[COL_DIV],
            color="tab:purple", linewidth=1.2, label="SHADE-10")

    if reinicios is not None and not reinicios.empty:
        for _, row in reinicios.iterrows():
            ev = row.get("evaluaciones_despues_reinicio", row.get("evaluaciones_antes_reinicio"))
            if pd.notna(ev):
                ax.axvline(x=float(ev), color="tab:purple", linewidth=0.7,
                           linestyle="--", alpha=0.6)
        # dummy para leyenda
        ax.axvline(x=-1, color="tab:purple", linewidth=0.7, linestyle="--",
                   alpha=0.6, label="Reinicio SHADE-10")

    ax.set_xlabel("Evaluaciones")
    ax.set_ylabel("Diversidad normalizada")
    ax.set_title(f"SHADE | CEC2017 f{funcid} — Diversidad: base vs SHADE-10 (semilla {semilla})")
    ax.legend(fontsize=8)
    ax.set_xlim(0, MAX_EVALS)
    plt.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"curva_diversidad_shade_base_vs_10_f{funcid}.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado en {outpath}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--funcid", type=int, default=10)
    parser.add_argument("--outdir", type=str,
                        default="memoria/figuras/reinicio")
    args = parser.parse_args()
    generar_figura(args.funcid, ROOT / args.outdir)


if __name__ == "__main__":
    main()
