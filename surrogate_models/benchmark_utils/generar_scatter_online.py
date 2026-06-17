"""
Scatter: evaluaciones evitadas (%) vs mejora relativa del error final por función.
X = candidatos_rechazados / candidatos_generados  (por función, agregado sobre semillas)
Y = (Base - RBF) / |Base|  (mejora relativa; positivo = surrogate mejor)
Color: verde=mejora, gris=empate, rojo=empeora  (umbral 1%)
"""
import json
import collections
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS = Path("results/cec")
TABLES  = Path("memoria/tablas/online")
OUT     = Path("memoria/figuras/surrogates_online")

MHS = {
    "AGE":   "online_final_surrogate_age",
    "DE":    "online_final_surrogate_de",
    "SHADE": "online_final_surrogate_shade",
}

THRESHOLD = 0.01
COLOR_WIN  = "#2ca02c"
COLOR_TIE  = "#7f7f7f"
COLOR_LOSS = "#d62728"


def load_eval_evitadas(folder: Path) -> dict:
    data = collections.defaultdict(lambda: {"rechazados": 0, "generados": 0})
    for f in folder.rglob("resumen_online.json"):
        fn = None
        for p in f.parts:
            if p.startswith("f") and p[1:].isdigit():
                fn = int(p[1:])
                break
        if fn is None:
            continue
        d = json.loads(f.read_text())
        data[fn]["rechazados"] += d.get("candidatos_rechazados", 0)
        data[fn]["generados"]  += d.get("candidatos_generados", 0)
    return {fn: v["rechazados"] / v["generados"] * 100 for fn, v in data.items() if v["generados"] > 0}


def load_error_diff(mh: str) -> dict:
    df = pd.read_csv(TABLES / f"online_medias_funcion_{mh.lower()}.csv")
    df = df[df["funcion"].str.match(r"^f\d+$", na=False)].copy()
    result = {}
    for _, row in df.iterrows():
        fn = int(row["funcion"].replace("f", ""))
        base = float(row["Base"])
        rbf  = float(row["RBF"])
        denom = abs(base) + 1e-300
        rel = 0.0 if base == 0 else (base - rbf) / denom
        result[fn] = rel
    return result


fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)

for ax, (mh, folder_name) in zip(axes, MHS.items()):
    folder = RESULTS / folder_name
    ev = load_eval_evitadas(folder)
    er = load_error_diff(mh)

    fns = sorted(set(ev) & set(er))
    x = np.array([ev[fn] for fn in fns])
    y = np.array([er[fn] for fn in fns])

    colors = []
    for v in y:
        if v > THRESHOLD:
            colors.append(COLOR_WIN)
        elif v < -THRESHOLD:
            colors.append(COLOR_LOSS)
        else:
            colors.append(COLOR_TIE)

    Y_MIN, Y_MAX = -2.2, 0.8
    y_plot = np.clip(y, Y_MIN, Y_MAX)
    is_clipped = (y < Y_MIN) | (y > Y_MAX)

    ax.scatter(x, y_plot, c=colors, s=60, zorder=3, edgecolors="none", alpha=0.85)
    for i, (xi, yi, c, clip) in enumerate(zip(x, y_plot, colors, is_clipped)):
        if clip:
            marker = "v" if y[i] < Y_MIN else "^"
            ax.scatter([xi], [yi], c=c, s=100, marker=marker, zorder=4,
                       edgecolors="black", linewidths=0.5)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_title(mh, fontsize=12)
    ax.set_xlabel("Evaluaciones evitadas (%)", fontsize=10)
    if mh == "AGE":
        ax.set_ylabel("Mejora relativa  (Base−RBF) / |Base|", fontsize=9)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=8)

    # label outliers
    for fn, xi, yi_raw, yi_plot, clip in zip(fns, x, y, y_plot, is_clipped):
        if clip:
            ax.annotate(f"F{fn:02d}", (xi, yi_plot),
                        textcoords="offset points", xytext=(4, 2), fontsize=6.5)

win_patch  = mpatches.Patch(color=COLOR_WIN,  label="Mejora (> 1 %)")
tie_patch  = mpatches.Patch(color=COLOR_TIE,  label="Empate (± 1 %)")
loss_patch = mpatches.Patch(color=COLOR_LOSS, label="Empeora (< −1 %)")
fig.legend(handles=[win_patch, tie_patch, loss_patch],
           loc="lower center", ncol=3, fontsize=9, frameon=False,
           bbox_to_anchor=(0.5, -0.03))

fig.suptitle("Evaluaciones evitadas vs impacto en el error final por función (RBF online)",
             fontsize=12, y=1.01)
fig.tight_layout()

OUT.mkdir(parents=True, exist_ok=True)
for ext in ["png", "pdf"]:
    path = OUT / f"scatter_online_mejora_por_funcion.{ext}"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Guardado: {path}")

plt.close()
