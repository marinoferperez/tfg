"""
Genera tabla LaTeX comparando la configuración homogénea (ε=1, nb=50) con la
configuración por algoritmo (ε=0.1, nb=25 para AGE y SHADE; idéntica para DE)
mostrando Spearman medio agregado sobre las 7 funciones de cribado y victorias.

Salida: memoria/tablas/cap06_ajuste_rbf/comparativa_homogenea_vs_peralgo.tex
"""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path("results/cec/cec2017_d10_tam50_reinicio_seleccionado")

PATH_HOMOGENEA = BASE / "benchmarking_surrogates_final_full/future_next/rbf_neighbors_50/rbf/no_acumulativo"
PATH_PERALGO   = BASE / "benchmarking_surrogates_offline_next/future_next/rbf/no_acumulativo"

OUT = Path("memoria/tablas/cap06_ajuste_rbf/comparativa_homogenea_vs_peralgo.tex")

FUNCS  = ["f1", "f4", "f10", "f12", "f18", "f22", "f29"]
ALGOS  = ["age", "shade"]
LABELS = {"age": "AGE", "shade": "SHADE"}


def load_spearman(base: Path, func: str, algo: str) -> float | None:
    p = base / func / algo / "rbf" / "rbf_metricas.json"
    if not p.exists():
        return None
    return float(json.loads(p.read_text())["spearman"])


def _fmt(v: float) -> str:
    return f"{v:.4f}"


def _bold(s: str) -> str:
    return f"\\textbf{{{s}}}"


def render() -> str:
    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        r"    \small",
        r"    \setlength{\tabcolsep}{5pt}",
        r"    \begin{tabular}{lrrr}",
        r"        \toprule",
        r"        \textbf{Algoritmo} "
        r"& \textbf{Homogénea} ($\varepsilon$=1, $k$=50) "
        r"& \textbf{Por algo} ($\varepsilon$=0.1, $k$=25) "
        r"& \textbf{Victorias homogénea} \\",
        r"        \midrule",
    ]

    for algo in ALGOS:
        vals_hom, vals_pa, wins = [], [], 0
        for func in FUNCS:
            h = load_spearman(PATH_HOMOGENEA, func, algo)
            p = load_spearman(PATH_PERALGO,   func, algo)
            if h is None or p is None:
                continue
            vals_hom.append(h)
            vals_pa.append(p)
            if h >= p:
                wins += 1

        if not vals_hom:
            lines.append(f"        {LABELS[algo]} & --- & --- & --- \\\\")
            continue

        mean_hom = sum(vals_hom) / len(vals_hom)
        mean_pa  = sum(vals_pa)  / len(vals_pa)
        hom_s = _fmt(mean_hom)
        pa_s  = _fmt(mean_pa)
        if mean_hom >= mean_pa:
            hom_s = _bold(hom_s)
        else:
            pa_s = _bold(pa_s)

        lines.append(
            f"        {LABELS[algo]} & {hom_s} & {pa_s} & {wins}/{len(FUNCS)} \\\\"
        )

    lines += [
        r"        \bottomrule",
        r"    \end{tabular}",
        r"    \caption{Comparativa de Spearman medio agregado sobre las 7 funciones de "
        r"cribado entre la configuración homogénea ($\varepsilon$=1, vecinos=50) y la "
        r"configuración por algoritmo ($\varepsilon$=0.1, vecinos=25). "
        r"Kernel \textit{multiquadric} y smoothing $10^{-3}$ fijos en ambas. "
        r"Negrita indica el mejor Spearman medio. "
        r"Para DE ambas configuraciones son idénticas.}",
        r"    \label{tab:comparativa_homogenea_vs_peralgo}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(), encoding="utf-8")
    print(f"Guardado en {OUT}")


if __name__ == "__main__":
    main()
