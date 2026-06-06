#!/usr/bin/env python3
"""Añade la diversidad ajustada por dimensión a los CSV de resultados CEC.

La columna original ``div_dist_euclidea_normalizada`` se conserva. La nueva
columna ``div_dist_euclidea_ajustada_dim`` representa Delta / D, de acuerdo
con el criterio de reinicio Delta / D < p * R.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = ROOT / "results" / "cec"
DEFAULT_DIMENSION = 10.0
DEFAULT_RANGO = 200.0

COL_DIVERSIDAD_BRUTA = "div_dist_euclidea"
COL_DIVERSIDAD_NORMALIZADA = "div_dist_euclidea_normalizada"
COL_DIVERSIDAD_AJUSTADA = "div_dist_euclidea_ajustada_dim"
COL_UMBRAL = "umbral_diversidad"
COL_UMBRAL_AJUSTADO = "umbral_diversidad_ajustada_dim"


def inferir_dimension(path: Path, default: float) -> float:
    for parte in reversed(path.parts):
        match = re.search(r"_d(\d+)(?:_|$)", parte)
        if match:
            return float(match.group(1))
    return float(default)


def parse_float(valor: str) -> float | None:
    texto = str(valor).strip()
    if texto == "":
        return None
    try:
        return float(texto)
    except ValueError:
        return None


def formatear(valor: float) -> str:
    return f"{valor:.12g}"


def insertar_columna_si_falta(fieldnames: list[str], nueva: str, despues_de: str) -> list[str]:
    if nueva in fieldnames:
        return fieldnames
    if despues_de in fieldnames:
        idx = fieldnames.index(despues_de) + 1
        return fieldnames[:idx] + [nueva] + fieldnames[idx:]
    return fieldnames + [nueva]


def recalcular_csv(path: Path, dimension_default: float, rango: float, dry_run: bool) -> bool:
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return False
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    tiene_bruta = COL_DIVERSIDAD_BRUTA in fieldnames
    tiene_norm = COL_DIVERSIDAD_NORMALIZADA in fieldnames
    tiene_umbral = COL_UMBRAL in fieldnames

    if not tiene_bruta and not tiene_norm and not tiene_umbral:
        return False

    dimension = inferir_dimension(path, dimension_default)
    nuevos_fieldnames = list(fieldnames)

    if tiene_bruta or tiene_norm:
        despues = COL_DIVERSIDAD_NORMALIZADA if tiene_norm else COL_DIVERSIDAD_BRUTA
        nuevos_fieldnames = insertar_columna_si_falta(
            nuevos_fieldnames,
            COL_DIVERSIDAD_AJUSTADA,
            despues,
        )

    if tiene_umbral:
        nuevos_fieldnames = insertar_columna_si_falta(
            nuevos_fieldnames,
            COL_UMBRAL_AJUSTADO,
            COL_UMBRAL,
        )

    for row in rows:
        if tiene_bruta:
            valor_bruto = parse_float(row.get(COL_DIVERSIDAD_BRUTA, ""))
            row[COL_DIVERSIDAD_AJUSTADA] = (
                "" if valor_bruto is None else formatear(valor_bruto / dimension)
            )
        elif tiene_norm:
            valor_norm = parse_float(row.get(COL_DIVERSIDAD_NORMALIZADA, ""))
            row[COL_DIVERSIDAD_AJUSTADA] = (
                "" if valor_norm is None else formatear(valor_norm * rango)
            )

        if tiene_umbral:
            umbral = parse_float(row.get(COL_UMBRAL, ""))
            row[COL_UMBRAL_AJUSTADO] = "" if umbral is None else formatear(umbral * rango)

    if dry_run:
        return True

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=nuevos_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Añade div_dist_euclidea_ajustada_dim = Delta / D a los CSV de CEC."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directorio raíz de resultados CEC.",
    )
    parser.add_argument("--dimension", type=float, default=DEFAULT_DIMENSION)
    parser.add_argument("--rango", type=float, default=DEFAULT_RANGO)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.results_dir.exists():
        raise FileNotFoundError(f"No existe el directorio: {args.results_dir}")

    rutas = set(args.results_dir.rglob("resultados_*.csv"))
    rutas.update(args.results_dir.rglob("reinicios_elitistas.csv"))

    procesados = 0
    for path in sorted(rutas):
        if recalcular_csv(path, args.dimension, args.rango, args.dry_run):
            procesados += 1

    modo = "se actualizarían" if args.dry_run else "actualizados"
    print(f"CSV {modo}: {procesados}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
