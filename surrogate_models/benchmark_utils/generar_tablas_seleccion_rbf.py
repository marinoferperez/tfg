"""
Genera tablas LaTeX para justificar la selección de hiperparámetros de RBF.

Tabla 1 (capítulo): comparación de kernels agregada sobre todos los algoritmos.
Tabla 2 (capítulo): configuración ganadora por algoritmo en el espacio reducido
                    (multiquadric, sm=0.001, nb in {25,50}).
Tablas 3-5 (apéndice): top-10 configuraciones por algoritmo en el espacio reducido.
"""
from __future__ import annotations

import ast
import csv
import re
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(
    "results/cec/cec2017_d10_tam50_reinicio_seleccionado/"
    "benchmarking_surrogates_hyperparameter_selection/future_next/rbf/no_acumulativo"
)
OUT_DIR = Path("memoria/tablas")

ALGORITHMS = ["age", "de", "shade"]
ALGO_LABELS = {"age": "AGE", "de": "DE", "shade": "SHADE"}
TOP_N_APENDICE = 10


def _clean_repr(raw: str) -> str:
    s = re.sub(r"np\.float64\(([^)]+)\)", r"\1", raw)
    s = re.sub(r"np\.int64\(([^)]+)\)", r"\1", s)
    s = re.sub(r"np\.nan\b", "None", s)
    return s


def _params_key(p: dict) -> tuple:
    return (
        p.get("kernel", ""),
        float(p.get("epsilon", 0)),
        float(p.get("smoothing", 0)),
        int(p.get("neighbors", 0)),
    )


def _rank_configs(configs: list[dict]) -> list[float]:
    scores = [
        c["score"] if c.get("error") is None and c.get("score") is not None else float("-inf")
        for c in configs
    ]
    sorted_unique = sorted(set(scores), reverse=True)
    rank_map: dict[float, float] = {}
    pos = 1
    for val in sorted_unique:
        count = scores.count(val)
        rank_map[val] = pos + (count - 1) / 2.0
        pos += count
    return [rank_map[s] for s in scores]


def load_and_aggregate() -> dict[str, list[dict]]:
    """Devuelve {algo: [{key, params, rank}]} ordenado por rank."""
    data: dict[str, dict[tuple, dict]] = {
        algo: defaultdict(lambda: {"ranks": [], "params": None})
        for algo in ALGORITHMS
    }
    for func_dir in sorted(BASE_DIR.iterdir()):
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
                ranks = _rank_configs(configs)
                for cfg, rank in zip(configs, ranks):
                    if cfg.get("error") is not None or cfg.get("score") is None:
                        continue
                    key = _params_key(cfg["params"])
                    entry = data[algo][key]
                    if entry["params"] is None:
                        entry["params"] = cfg["params"]
                    entry["ranks"].append(rank)

    result = {}
    for algo, configs in data.items():
        rows = []
        for key, entry in configs.items():
            if not entry["ranks"] or entry["params"] is None:
                continue
            rows.append({
                "key": key,
                "params": entry["params"],
                "rank": round(sum(entry["ranks"]) / len(entry["ranks"]), 3),
            })
        rows.sort(key=lambda r: r["rank"])
        for pos, r in enumerate(rows, start=1):
            r["pos"] = pos
        result[algo] = rows
    return result


def _bold(s: str) -> str:
    return r"\textbf{" + s + r"}"


def _kernel_tex(k: str) -> str:
    return r"\textit{mq}" if k == "multiquadric" else r"\textit{gauss}"


def _eps_tex(e: float) -> str:
    return f"${e}$"


def _sm_tex(s: float) -> str:
    if abs(s - 0.001) < 1e-9:
        return r"$10^{-3}$"
    if abs(s - 0.01) < 1e-9:
        return r"$10^{-2}$"
    return f"${s}$"


# ─── Tabla 1: comparación de kernels ────────────────────────────────────────

def tabla_kernels(agg: dict[str, list[dict]]) -> str:
    # Rank medio por (algo, kernel)
    from collections import defaultdict
    data = defaultdict(list)
    for algo, rows in agg.items():
        for r in rows:
            k = r["params"].get("kernel", "")
            data[(algo, k)].append(r["rank"])

    kernels = ["multiquadric", "gaussian"]
    algos = ALGORITHMS

    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        r"    \small",
        r"    \setlength{\tabcolsep}{5pt}",
        r"    \begin{tabular}{lrrr|r}",
        r"        \toprule",
        r"        \textbf{Kernel} & \textbf{AGE} & \textbf{DE} & \textbf{SHADE} & \textbf{Rank medio global} \\",
        r"        \midrule",
    ]

    global_means = {}
    for k in kernels:
        vals_per_algo = []
        cols = []
        for algo in algos:
            v = data.get((algo, k), [])
            mean = round(sum(v) / len(v), 3) if v else None
            cols.append(f"{mean:.3f}" if mean is not None else "---")
            if mean is not None:
                vals_per_algo.append(mean)
        global_mean = round(sum(vals_per_algo) / len(vals_per_algo), 3) if vals_per_algo else None
        global_means[k] = global_mean
        lines.append(
            f"        {_kernel_tex(k)} & {' & '.join(cols)} & {global_mean:.3f} \\\\"
        )

    # Negrita en el mejor (menor) por columna
    # Reescribir filas con negrita
    best_global = min(global_means.values())
    # Reconstruir con negrita
    lines_data = []
    for k in kernels:
        vals_per_algo = []
        cols = []
        for algo in algos:
            v = data.get((algo, k), [])
            mean = round(sum(v) / len(v), 3) if v else None
            vals_per_algo.append(mean)
        global_mean = global_means[k]

        best_per_algo = {
            algo: min(
                round(sum(data.get((algo2, k2), [1e9])) / max(len(data.get((algo2, k2), [1])), 1), 3)
                for k2 in kernels
            )
            for algo in algos
            for algo2 in [algo]
        }

        col_strs = []
        for algo, mean in zip(algos, vals_per_algo):
            s = f"{mean:.3f}" if mean is not None else "---"
            best = min(
                round(sum(data.get((algo, k2), [1e9])) / max(len(data.get((algo, k2), [1])), 1), 3)
                for k2 in kernels
            )
            if mean is not None and abs(mean - best) < 1e-9:
                s = _bold(s)
            col_strs.append(s)

        gm_str = f"{global_mean:.3f}"
        if abs(global_mean - best_global) < 1e-9:
            gm_str = _bold(gm_str)

        lines_data.append(
            f"        {_kernel_tex(k)} & {' & '.join(col_strs)} & {gm_str} \\\\"
        )

    # Reemplazar las líneas de datos
    lines = lines[:-2] + lines_data

    lines += [
        r"        \bottomrule",
        r"    \end{tabular}",
        r"    \caption{Rank medio por kernel agregado sobre todas las configuraciones candidatas "
        r"y algoritmos. Negrita indica el mejor valor por columna (menor rank = mejor).}",
        r"    \label{tab:rbf_seleccion_kernel}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# ─── Tabla 2: selección final por algoritmo ──────────────────────────────────

def tabla_seleccion_final(agg: dict[str, list[dict]]) -> str:
    # Espacio reducido: multiquadric, sm=0.001, nb in {25, 50}
    selected = {}
    for algo, rows in agg.items():
        candidates = [
            r for r in rows
            if r["params"].get("kernel") == "multiquadric"
            and abs(float(r["params"].get("smoothing", 0)) - 0.001) < 1e-9
            and int(r["params"].get("neighbors", 0)) in (25, 50)
        ]
        if candidates:
            selected[algo] = candidates[0]  # mejor (ya ordenado por rank)

    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        r"    \small",
        r"    \setlength{\tabcolsep}{5pt}",
        r"    \begin{tabular}{lrrrl}",
        r"        \toprule",
        r"        \textbf{Algoritmo} & \textbf{$\varepsilon$} & \textbf{Vecinos} "
        r"& \textbf{Rank medio} & \textbf{Pos.} \\",
        r"        \midrule",
    ]

    for algo in ALGORITHMS:
        r = selected[algo]
        p = r["params"]
        lines.append(
            f"        {ALGO_LABELS[algo]} & {_eps_tex(float(p['epsilon']))} "
            f"& {int(p['neighbors'])} & {r['rank']} & {r['pos']}\\textsuperscript{{a}} \\\\"
        )

    lines += [
        r"        \bottomrule",
        r"    \end{tabular}",
        r"    \footnotesize\textsuperscript{a}Posición dentro del ranking completo de 36 configuraciones.",
        r"    \caption{Configuración de RBF seleccionada por algoritmo. Kernel \textit{multiquadric} "
        r"y smoothing $10^{-3}$ fijos para todos.}",
        r"    \label{tab:rbf_seleccion_final}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# ─── Tablas apéndice: top-N por algoritmo en espacio reducido ────────────────

def tabla_apendice_algo(agg: dict[str, list[dict]], algo: str, top_n: int) -> str:
    candidates = [
        r for r in agg[algo]
        if r["params"].get("kernel") == "multiquadric"
        and abs(float(r["params"].get("smoothing", 0)) - 0.001) < 1e-9
        and int(r["params"].get("neighbors", 0)) in (25, 50, 100)
    ]
    top = candidates[:top_n]
    label = ALGO_LABELS[algo]

    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        r"    \small",
        r"    \setlength{\tabcolsep}{5pt}",
        r"    \begin{tabular}{rrr|rr}",
        r"        \toprule",
        r"        \textbf{$\varepsilon$} & \textbf{Vecinos} & \textbf{Smoothing} "
        r"& \textbf{Rank medio} & \textbf{Pos.} \\",
        r"        \midrule",
    ]

    best_rank = top[0]["rank"] if top else None
    for r in top:
        p = r["params"]
        rank_s = f"{r['rank']}"
        if best_rank is not None and abs(r["rank"] - best_rank) < 1e-9:
            rank_s = _bold(rank_s)
        lines.append(
            f"        {_eps_tex(float(p['epsilon']))} & {int(p['neighbors'])} "
            f"& {_sm_tex(float(p['smoothing']))} & {rank_s} & {r['pos']} \\\\"
        )

    lines += [
        r"        \bottomrule",
        r"    \end{tabular}",
        f"    \\caption{{Top-{top_n} configuraciones de RBF para {label} en el espacio reducido "
        r"(kernel \textit{multiquadric}, vecinos $\in\{25, 50, 100\}$). "
        r"Pos.\ indica la posición en el ranking completo de 36 configuraciones.}",
        f"    \\label{{tab:rbf_apendice_{algo}}}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    print("Cargando datos...")
    agg = load_and_aggregate()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Tabla 1: kernels
    p = OUT_DIR / "rbf_seleccion_kernel.tex"
    p.write_text(tabla_kernels(agg), encoding="utf-8")
    print(f"  → {p}")

    # Tabla 2: selección final
    p = OUT_DIR / "rbf_seleccion_final.tex"
    p.write_text(tabla_seleccion_final(agg), encoding="utf-8")
    print(f"  → {p}")

    # Tablas apéndice
    for algo in ALGORITHMS:
        p = OUT_DIR / f"rbf_apendice_{algo}.tex"
        p.write_text(tabla_apendice_algo(agg, algo, TOP_N_APENDICE), encoding="utf-8")
        print(f"  → {p}")


if __name__ == "__main__":
    main()
