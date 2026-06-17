"""
Genera la tabla LaTeX de ranking global de configuraciones RBF para el apéndice D.
Ordena por 'Peor pos.' (criterio minimax) y destaca la configuración seleccionada.
Salida: memoria/tablas/cap06_ajuste_rbf/apendice/rbf_ranking_global.tex
"""
from pathlib import Path
import pandas as pd

IN = Path("memoria/tablas/ajuste_configuraciones_rbf.xlsx")
OUT = Path("memoria/tablas/cap06_ajuste_rbf/apendice/rbf_ranking_global.tex")

SELECTED = {"kernel": "multiquadric", "epsilon": 1.0, "smoothing": 0.001, "neighbors": 50}

def kernel_tex(k: str) -> str:
    return r"\textit{mq}" if k == "multiquadric" else r"\textit{gauss}"

def eps_tex(v: float) -> str:
    if v == 0.1:   return r"$0.1$"
    if v == 1.0:   return r"$1.0$"
    if v == 10.0:  return r"$10.0$"
    return f"${v}$"

def sm_tex(v: float) -> str:
    if abs(v - 0.001) < 1e-9: return r"$10^{-3}$"
    if abs(v - 0.01)  < 1e-9: return r"$10^{-2}$"
    return f"${v}$"

def bold(s: str) -> str:
    return r"\textbf{" + s + r"}"

def is_selected(row) -> bool:
    return (
        row["Kernel"] == SELECTED["kernel"]
        and abs(float(row["ε"]) - SELECTED["epsilon"]) < 1e-9
        and abs(float(row["Smoothing"]) - SELECTED["smoothing"]) < 1e-9
        and int(row["Vecinos"]) == SELECTED["neighbors"]
    )

df = pd.read_excel(IN, sheet_name="Resumen global")
df = df.sort_values("Peor pos.").reset_index(drop=True)
total = len(df)
df = df.head(15)

lines = [
    r"\begin{table}[H]",
    r"    \centering",
    r"    \small",
    r"    \setlength{\tabcolsep}{5pt}",
    r"    \begin{adjustbox}{max width=\textwidth}",
    r"    \begin{tabular}{lrrr|rrr|r}",
    r"        \toprule",
    r"        \textbf{Kernel} & \textbf{$\varepsilon$} & \textbf{Vecinos} & \textbf{Smoothing}"
    r" & \textbf{Pos.\ AGE} & \textbf{Pos.\ DE} & \textbf{Pos.\ SHADE} & \textbf{Peor pos.} \\",
    r"        \midrule",
]

for _, row in df.iterrows():
    sel = is_selected(row)
    k   = kernel_tex(row["Kernel"])
    eps = eps_tex(float(row["ε"]))
    sm  = sm_tex(float(row["Smoothing"]))
    nb  = str(int(row["Vecinos"]))
    pa  = str(int(row["Pos. AGE"]))
    pd_ = str(int(row["Pos. DE"]))
    ps  = str(int(row["Pos. SHADE"]))
    pp  = str(int(row["Peor pos."]))
    if sel:
        k, eps, sm, nb, pa, pd_, ps, pp = [bold(x) for x in [k, eps, sm, nb, pa, pd_, ps, pp]]
    lines.append(f"        {k} & {eps} & {nb} & {sm} & {pa} & {pd_} & {ps} & {pp} \\\\")

lines += [
    r"        \bottomrule",
    r"    \end{tabular}",
    r"    \end{adjustbox}",
    f"    \\caption{{Ranking global de las {total} configuraciones RBF candidatas, "
    r"ordenadas por la peor posición obtenida entre las tres metaheurísticas (criterio minimax). "
    r"La configuración seleccionada como homogénea aparece en negrita.}",
    r"    \label{tab:rbf_ranking_global}",
    r"\end{table}",
]

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Escrito: {OUT}")
