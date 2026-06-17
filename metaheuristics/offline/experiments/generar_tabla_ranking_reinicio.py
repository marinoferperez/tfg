"""
Genera las tablas LaTeX de ranking TACOLAB para las configuraciones de reinicio
de AGE, DE y SHADE. Columnas: Configuración | Rank medio (100%) | Victorias.

Lee los ficheros ya generados por TACOLAB en resultados/tacolab_reinicio/:
  - ranking_{algo}_tacolab.tex  → rank medio al 100%
  - media_{algo}_tacolab.tex    → fila "Best" con victorias por configuración

Salida: memoria/tablas/tacolab/{age,de,shade}_tacolab_ranking.tex
"""
from __future__ import annotations
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "resultados/tacolab_reinicio"
OUT = ROOT / "memoria/tablas/tacolab"

ALGORITMOS = {
    "age":   ("AGE",   {
        "base":        "AGE",
        "reinicio-1":  "AGE-1",
        "reinicio-3":  "AGE-3",
        "reinicio-5":  "AGE-5",
        "reinicio-7":  "AGE-7",
        "reinicio-10": "AGE-10",
    }),
    "de":    ("DE",    {
        "base":        "DE",
        "reinicio-1":  "DE-1",
        "reinicio-3":  "DE-3",
        "reinicio-5":  "DE-5",
        "reinicio-7":  "DE-7",
        "reinicio-10": "DE-10",
    }),
    "shade": ("SHADE", {
        "base":        "SHADE",
        "reinicio-1":  "SHADE-1",
        "reinicio-3":  "SHADE-3",
        "reinicio-5":  "SHADE-5",
        "reinicio-7":  "SHADE-7",
        "reinicio-10": "SHADE-10",
    }),
}


def parse_ranking(tex: str) -> dict[str, float]:
    """Extrae {sheet_name: rank_medio_al_100%} del tex de ranking."""
    lines = tex.splitlines()

    # Encuentra la línea de cabecera que contiene los milestones
    header_idx = None
    headers = []
    for i, line in enumerate(lines):
        if re.search(r"\b100\b", line) and "&" in line:
            header_idx = i
            cells = [c.strip() for c in line.replace("\\\\", "").split("&")]
            headers = cells
            break

    if header_idx is None:
        raise ValueError("No se encontró cabecera con milestone 100.")

    # Índice de la columna 100 (puede ser la única o una de varias)
    col_100 = None
    for j, h in enumerate(headers):
        if h.strip() == "100":
            col_100 = j
            break
    if col_100 is None:
        raise ValueError("No se encontró columna 100 en la cabecera.")

    result = {}
    for line in lines[header_idx + 1:]:
        line_clean = line.strip()
        if line_clean.startswith("\\") or not line_clean or line_clean.startswith("{"):
            continue
        if "toprule" in line_clean or "midrule" in line_clean or "bottomrule" in line_clean:
            continue
        cells = [c.strip() for c in line_clean.replace("\\\\", "").split("&")]
        if len(cells) <= col_100:
            continue
        key = cells[0].strip()
        try:
            result[key] = float(cells[col_100])
        except ValueError:
            continue
    return result


def parse_victorias(tex: str) -> dict[str, int]:
    """Extrae {sheet_name: victorias} de la fila Best del tex de medias."""
    # Header line: {} & base & reinicio-1 & reinicio-10 & reinicio-3 & reinicio-5 & reinicio-7
    # Best line:   Best &  3  &  23  &  1  &  4  &  0  &  0  \\
    header_line = None
    best_line = None
    for line in tex.splitlines():
        if "base" in line and "reinicio" in line:
            header_line = line
        if line.strip().startswith("Best"):
            best_line = line
    if not header_line or not best_line:
        raise ValueError("No se encontró la cabecera o la fila Best en el tex de medias.")

    def split_cells(line):
        line = line.replace("\\\\", "").strip()
        return [c.strip() for c in line.split("&")]

    headers = split_cells(header_line)  # ['{}', 'base', 'reinicio-1', ...]
    values = split_cells(best_line)     # ['Best', '3', '23', ...]

    result = {}
    for h, v in zip(headers[1:], values[1:]):
        try:
            result[h] = int(v)
        except ValueError:
            result[h] = 0
    return result


def generar_tex(labels_map: dict[str, str], ranking: dict[str, float],
                victorias: dict[str, int]) -> str:
    # Construye filas combinando rank + wins, ordena por rank
    filas = []
    for sheet, label in labels_map.items():
        r = ranking.get(sheet, float("inf"))
        v = victorias.get(sheet, 0)
        filas.append((label, r, v))
    filas.sort(key=lambda x: x[1])

    best_rank = min(f[1] for f in filas)
    best_wins = max(f[2] for f in filas)

    lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"\textbf{Configuración} & \textbf{Rank medio (100\%)} & \textbf{Victorias} \\",
        r"\midrule",
    ]
    for label, rank, wins in filas:
        rank_str = f"{rank:.3f}"
        if abs(rank - best_rank) < 1e-9:
            rank_str = r"\textbf{" + rank_str + "}"
        wins_str = str(wins)
        if wins == best_wins:
            wins_str = r"\textbf{" + wins_str + "}"
        lines.append(f"{label} & {rank_str} & {wins_str} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for algo, (_, labels_map) in ALGORITMOS.items():
        ranking_tex = (SRC / f"ranking_{algo}_tacolab.tex").read_text(encoding="utf-8")
        media_tex = (SRC / f"media_{algo}_tacolab.tex").read_text(encoding="utf-8")

        ranking = parse_ranking(ranking_tex)
        victorias = parse_victorias(media_tex)

        tex = generar_tex(labels_map, ranking, victorias)
        out = OUT / f"{algo}_tacolab_ranking.tex"
        out.write_text(tex, encoding="utf-8")
        print(f"Escrito: {out}")


if __name__ == "__main__":
    main()
