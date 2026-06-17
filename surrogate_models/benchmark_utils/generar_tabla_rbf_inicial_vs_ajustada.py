"""
Genera la tabla de rendimiento offline del RBF final sobre CEC2017 completo.

Config final: benchmarking_surrogates_offline_rbf_rsm_tuned/future_next/rbf/no_acumulativo
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PATH_FINAL = ROOT / "results/cec/cec2017_d10_tam50_reinicio_seleccionado/benchmarking_surrogates_offline_rbf_rsm_tuned/future_next/rbf/no_acumulativo"

OUT_MAIN = ROOT / "memoria/tablas/comparativa_rbf_inicial_ajustada_bloques.tex"

ALGOS  = ["age", "de", "shade"]
LABELS = {"age": "AGE", "de": "DE", "shade": "SHADE"}
FUNCS  = [f"f{i}" for i in range(1, 31)]


def load_metrics(base: Path, func: str, algo: str) -> dict | None:
    p = base / func / algo / "rbf" / "rbf_metricas.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def aggregate(base: Path, algo: str) -> dict:
    """Agrega métricas sobre las 30 funciones para un algoritmo."""
    spearman_vals, nrmse_vals, nmae_vals = [], [], []
    train_vals, pred_vals = [], []

    for func in FUNCS:
        m = load_metrics(base, func, algo)
        if m is None:
            continue
        spearman_vals.append(m["spearman"])
        nrmse_vals.append(m["nrmse"])
        nmae_vals.append(m["nmae"])
        train_vals.append(m["train_time_s"])
        pred_vals.append(m["predict_time_s"])

    n = len(spearman_vals)
    return {
        "n_funcs": n,
        "spearman": sum(spearman_vals) / n,
        "nrmse":    sum(nrmse_vals) / n,
        "nmae":     sum(nmae_vals) / n,
        "train_s":  sum(train_vals) / n,
        "pred_s":   sum(pred_vals) / n,
    }


def fmt(v: float, decimals: int = 4) -> str:
    return f"{v:.{decimals}f}"


# ─── Tabla principal ──────────────────────────────────────────────────────────

def gen_main_table() -> str:
    rows = []
    for algo in ALGOS:
        metrics = aggregate(PATH_FINAL, algo)
        rows.append((LABELS[algo], metrics))

    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        r"    \small",
        r"    \setlength{\tabcolsep}{5pt}",
        r"    \begin{adjustbox}{max width=\textwidth}",
        r"    \begin{tabular}{lr}",
        r"        \toprule",
        r"        \textbf{Algoritmo} & \textbf{Spearman medio} \\",
        r"        \midrule",
    ]

    for label, metrics in rows:
        lines.append(f"        {label} & {fmt(metrics['spearman'])} \\\\")

    lines += [
        r"        \bottomrule",
        r"    \end{tabular}",
        r"    \end{adjustbox}",
        r"    \caption{Rendimiento \textit{offline} de RBF con la configuración homogénea final "
        r"($\varepsilon=1.0$, \textit{smoothing}=$10^{-3}$, 50 vecinos, kernel "
        r"\textit{multiquadric} y $\text{degree}=-1$) sobre CEC2017 completo "
        r"(30 funciones, 51 semillas).}",
        r"    \label{tab:rendimiento_rbf_final_offline}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    main_tex = gen_main_table()
    OUT_MAIN.write_text(main_tex, encoding="utf-8")
    print(f"Tabla principal: {OUT_MAIN}")


if __name__ == "__main__":
    main()
