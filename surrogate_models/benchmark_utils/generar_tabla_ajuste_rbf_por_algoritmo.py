"""
Genera tablas LaTeX del ajuste de hiperparámetros de RBF separadas por algoritmo.

Para cada algoritmo (AGE, DE, SHADE) produce una tabla con las N mejores
configuraciones de RBF ordenadas por rango medio sobre Spearman interno.
El rango se calcula por cada combinación (función × bloque × semilla):
las 36 configuraciones se ordenan por Spearman y se asigna rango 1 a la mejor.
La tabla muestra rango medio, Spearman medio y tiempos de entrenamiento y predicción.
"""
from __future__ import annotations

import argparse
import ast
import csv
import re
from collections import defaultdict
from pathlib import Path

DEFAULT_BASE_DIR = (
    "results/cec/cec2017_d10_tam50_reinicio_seleccionado/"
    "benchmarking_surrogates_hyperparameter_selection/future_next/rbf/no_acumulativo"
)
DEFAULT_OUT_DIR = "memoria/tablas"
DEFAULT_TOP_N = 5

ALGORITHMS = ["age", "de", "shade"]
ALGO_LABELS = {"age": "AGE", "de": "DE", "shade": "SHADE"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help="Número de configuraciones a mostrar por tabla.",
    )
    parser.add_argument(
        "--prefix",
        default="ajuste_interno_rbf",
        help="Prefijo de los ficheros de salida.",
    )
    return parser.parse_args()


def _clean_repr(raw: str) -> str:
    s = re.sub(r"np\.float64\(([^)]+)\)", r"\1", raw)
    s = re.sub(r"np\.int64\(([^)]+)\)", r"\1", s)
    s = re.sub(r"np\.nan\b", "None", s)
    return s


def _params_key(params: dict) -> tuple:
    return (
        params.get("kernel", ""),
        float(params.get("epsilon", 0)),
        float(params.get("smoothing", 0)),
        int(params.get("neighbors", 0)),
        int(params.get("degree", -1)),
    )


def _rank_configs(configs: list[dict]) -> list[float]:
    """
    Asigna rango 1 a la config con mayor Spearman. En caso de empate, rango promedio.
    Devuelve lista de rangos en el mismo orden que configs.
    """
    scores = [c["score"] if c.get("error") is None and c.get("score") is not None else float("-inf")
              for c in configs]
    n = len(scores)
    # Para cada posición, su rango = 1 + número de configs con score estrictamente mayor
    # En empates, rango promedio de las posiciones que ocuparían
    sorted_unique = sorted(set(scores), reverse=True)
    rank_map: dict[float, float] = {}
    pos = 1
    for val in sorted_unique:
        count = scores.count(val)
        avg_rank = pos + (count - 1) / 2.0
        rank_map[val] = avg_rank
        pos += count
    return [rank_map[s] for s in scores]


def load_data(base_dir: Path) -> dict[str, dict[tuple, dict]]:
    """
    Returns {algorithm: {params_key: {"ranks": [...], "spearman": [...],
                                       "train_time_s": [...], "predict_time_s": [...],
                                       "params": dict}}}
    Los rangos se calculan por fila (función × bloque × semilla).
    """
    data: dict[str, dict[tuple, dict]] = {
        algo: defaultdict(lambda: {
            "ranks": [], "spearman": [], "train_time_s": [], "predict_time_s": [],
            "params": None,
        })
        for algo in ALGORITHMS
    }

    for func_dir in sorted(base_dir.iterdir()):
        if not func_dir.is_dir() or not func_dir.name.startswith("f"):
            continue
        for algo in ALGORITHMS:
            runs_path = func_dir / algo / "rbf" / "rbf_runs.csv"
            if not runs_path.exists():
                continue
            rows = list(csv.DictReader(runs_path.read_text(encoding="utf-8").splitlines()))
            for row in rows:
                raw = row.get("tuning_resultados", "")
                if not raw:
                    continue
                try:
                    configs = ast.literal_eval(_clean_repr(raw))
                except Exception:
                    continue
                if not configs:
                    continue

                ranks = _rank_configs(configs)
                for cfg, rank in zip(configs, ranks):
                    if cfg.get("error") is not None or cfg.get("score") is None:
                        continue
                    key = _params_key(cfg["params"])
                    entry = data[algo][key]
                    if entry["params"] is None:
                        entry["params"] = cfg["params"]
                    entry["ranks"].append(rank)
                    entry["spearman"].append(float(cfg["score"]))
                    entry["train_time_s"].append(float(cfg.get("train_time_s") or 0))
                    entry["predict_time_s"].append(float(cfg.get("predict_time_s") or 0))
    return data


def aggregate(data: dict[str, dict[tuple, dict]]) -> dict[str, list[dict]]:
    """Promedia métricas por (algorithm, params_key) y ordena por rango medio asc."""
    result: dict[str, list[dict]] = {}
    for algo, configs in data.items():
        rows = []
        for key, entry in configs.items():
            n = len(entry["ranks"])
            if n == 0 or entry["params"] is None:
                continue
            rows.append(
                {
                    "params": entry["params"],
                    "rank": sum(entry["ranks"]) / n,
                    "spearman": sum(entry["spearman"]) / n,
                    "train_time_s": sum(entry["train_time_s"]) / n,
                    "predict_time_s": sum(entry["predict_time_s"]) / n,
                    "n": n,
                }
            )
        rows.sort(key=lambda r: r["rank"])
        result[algo] = rows
    return result


def _fmt(v: float, decimals: int = 4) -> str:
    return f"{v:.{decimals}f}"


def _bold(s: str) -> str:
    return f"\\textbf{{{s}}}"


def _epsilon_tex(eps: float) -> str:
    if abs(eps - 0.1) < 1e-9:
        return r"$0.1$"
    if abs(eps - 1.0) < 1e-9:
        return r"$1.0$"
    if abs(eps - 10.0) < 1e-9:
        return r"$10.0$"
    return f"${eps}$"


def _smoothing_tex(sm: float) -> str:
    if abs(sm - 0.001) < 1e-9:
        return r"$10^{-3}$"
    if abs(sm - 0.01) < 1e-9:
        return r"$10^{-2}$"
    return f"${sm}$"


def _kernel_tex(kernel: str) -> str:
    return r"\textit{mq}" if kernel == "multiquadric" else r"\textit{gauss}"


def render_table(rows: list[dict], top_n: int, algo_label: str) -> str:
    top = rows[:top_n]
    kernels = {r["params"].get("kernel") for r in top}
    has_mixed_kernels = len(kernels) > 1

    best_rank = min(r["rank"] for r in top)
    best_sp = max(r["spearman"] for r in top)
    best_train = min(r["train_time_s"] for r in top)
    best_pred = min(r["predict_time_s"] for r in top)

    if has_mixed_kernels:
        col_spec = r"llll|rrrr"
        header_line = (
            r"            \textbf{Kernel} & \textbf{$\varepsilon$} & \textbf{Smoothing} "
            r"& \textbf{Vecinos} & \textbf{Rank medio} & \textbf{Spearman} "
            r"& \textbf{Entren.\ (s)} & \textbf{Pred.\ (s)} \\"
        )
    else:
        col_spec = r"lll|rrrr"
        header_line = (
            r"            \textbf{$\varepsilon$} & \textbf{Smoothing} & \textbf{Vecinos} "
            r"& \textbf{Rank medio} & \textbf{Spearman} "
            r"& \textbf{Entren.\ (s)} & \textbf{Pred.\ (s)} \\"
        )

    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        r"    \small",
        r"    \setlength{\tabcolsep}{4pt}",
        r"    \begin{adjustbox}{max width=\textwidth}",
        f"        \\begin{{tabular}}{{{col_spec}}}",
        r"            \toprule",
        header_line,
        r"            \midrule",
    ]

    for r in top:
        p = r["params"]
        kernel = p.get("kernel", "multiquadric")
        eps = float(p.get("epsilon", 1.0))
        sm = float(p.get("smoothing", 0.001))
        nb = int(p.get("neighbors", 0))

        rk_s = _fmt(r["rank"], 3)
        sp_s = _fmt(r["spearman"])
        tr_s = _fmt(r["train_time_s"])
        pr_s = _fmt(r["predict_time_s"])

        if abs(r["rank"] - best_rank) < 1e-9:
            rk_s = _bold(rk_s)
        if abs(r["spearman"] - best_sp) < 1e-9:
            sp_s = _bold(sp_s)
        if abs(r["train_time_s"] - best_train) < 1e-9:
            tr_s = _bold(tr_s)
        if abs(r["predict_time_s"] - best_pred) < 1e-9:
            pr_s = _bold(pr_s)

        if has_mixed_kernels:
            lines.append(
                f"            {_kernel_tex(kernel)} & {_epsilon_tex(eps)} "
                f"& {_smoothing_tex(sm)} & {nb} "
                f"& {rk_s} & {sp_s} & {tr_s} & {pr_s} \\\\"
            )
        else:
            lines.append(
                f"            {_epsilon_tex(eps)} & {_smoothing_tex(sm)} & {nb} "
                f"& {rk_s} & {sp_s} & {tr_s} & {pr_s} \\\\"
            )

    algo_lower = algo_label.lower()
    common_kernel = list(kernels)[0] if not has_mixed_kernels else None
    if common_kernel == "multiquadric":
        kernel_note = r"todas con $\text{kernel}=\textit{multiquadric}$ y $\text{degree}=-1$"
    elif has_mixed_kernels:
        kernel_note = (
            r"$\text{degree}=-1$ en todas; "
            r"\textit{mq} = \textit{multiquadric}, \textit{gauss} = \textit{gaussian}"
        )
    else:
        kernel_note = r"$\text{degree}=-1$"

    lines += [
        r"            \bottomrule",
        r"        \end{tabular}",
        r"    \end{adjustbox}",
        (
            f"    \\caption[Cinco mejores configuraciones de RBF para {algo_label} "
            f"según rango medio sobre Spearman interno.]"
            f"{{Cinco mejores configuraciones de RBF para {algo_label}. "
            f"El rango medio se calcula por combinación función$\\times$bloque$\\times$semilla "
            f"sobre las 36 configuraciones candidatas ({kernel_note}).}}"
        ),
        f"    \\label{{tab:ajuste_interno_rbf_{algo_lower}}}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cargando datos desde {base_dir} ...")
    raw = load_data(base_dir)
    agg = aggregate(raw)

    for algo in ALGORITHMS:
        rows = agg.get(algo, [])
        if not rows:
            print(f"  {algo}: sin datos, omitido.")
            continue
        label = ALGO_LABELS[algo]
        tex = render_table(rows, args.top_n, label)
        out_path = out_dir / f"{args.prefix}_{algo}.tex"
        out_path.write_text(tex, encoding="utf-8")
        print(f"\n  {label}: {out_path}")
        print(f"    {'Kernel':>14} {'eps':>5} {'sm':>7} {'nb':>4} {'Rank':>8} {'Spearman':>10} {'Train':>8} {'Pred':>8}")
        for r in rows[: args.top_n]:
            p = r["params"]
            print(
                f"    {p['kernel']:>14} {p['epsilon']:>5} {p['smoothing']:>7} {p['neighbors']:>4} "
                f"{r['rank']:>8.3f} {r['spearman']:>10.4f} "
                f"{r['train_time_s']:>8.4f} {r['predict_time_s']:>8.4f}"
            )


if __name__ == "__main__":
    main()
