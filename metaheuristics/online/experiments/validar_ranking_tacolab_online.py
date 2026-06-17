from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results/cec"
TACO = ROOT / "resultados/taco_online"

ALGORITMOS = {
    "age": ("ao", "online_final_base_age", "online_final_surrogate_age"),
    "de": ("do", "online_final_base_de", "online_final_surrogate_de"),
    "shade": ("so", "online_final_base_shade", "online_final_surrogate_shade"),
}
TOL_RANK = 1e-12


def normalizar_cero(valor: float) -> float:
    return 0.0 if abs(float(valor)) <= TOL_RANK else float(valor)


def cargar_resultado(root_name: str, prefix: str, funcid: int) -> pd.DataFrame:
    func_dir = RESULTS / root_name / f"f{funcid}"
    files = list(func_dir.glob(f"results_{prefix}_*/results_{funcid}_10.txt"))
    if len(files) != 1:
        raise RuntimeError(f"No se pudo resolver un único fichero para {root_name} f{funcid}: {files}")
    df = pd.read_csv(files[0])
    return df.groupby("milestone", as_index=False)["error"].mean()


def calcular_ranking(algoritmo: str) -> pd.DataFrame:
    prefix, base_root, surrogate_root = ALGORITMOS[algoritmo]
    rows = []
    for funcid in range(1, 31):
        base = cargar_resultado(base_root, prefix, funcid).rename(columns={"error": "base"})
        surrogate = cargar_resultado(surrogate_root, prefix, funcid).rename(columns={"error": "surrogate"})
        merged = base.merge(surrogate, on="milestone")
        for _, row in merged.iterrows():
            rows.append(
                {
                    "funcid": funcid,
                    "milestone": int(row["milestone"]),
                    "base": normalizar_cero(float(row["base"])),
                    "surrogate": normalizar_cero(float(row["surrogate"])),
                }
            )
    df = pd.DataFrame(rows)
    out = []
    for milestone, group in df.groupby("milestone"):
        ranks = group[["base", "surrogate"]].rank(axis=1, method="average", ascending=True)
        out.append(
            {
                "milestone": int(milestone),
                "base": float(ranks["base"].mean()),
                "surrogate": float(ranks["surrogate"].mean()),
            }
        )
    return pd.DataFrame(out).sort_values("milestone")


def cargar_taco_ranking(algoritmo: str) -> pd.DataFrame:
    text = (TACO / f"ranking_{algoritmo}_online.tex").read_text(encoding="utf-8")
    header_match = re.search(r"\{\}\s*&(.+?)\\\\", text)
    if not header_match:
        raise RuntimeError(f"No se pudo leer la cabecera TACOLAB de {algoritmo}")
    milestones = [int(x) for x in re.findall(r"\d+", header_match.group(1))]

    rows = []
    for name in ("base", "surrogate"):
        row_match = re.search(rf"^{name}\s*&(.+?)\\\\", text, re.MULTILINE)
        if not row_match:
            raise RuntimeError(f"No se pudo leer la fila {name} de {algoritmo}")
        values = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", row_match.group(1))]
        rows.append(pd.DataFrame({"milestone": milestones, name: values}))
    return rows[0].merge(rows[1], on="milestone")


def main() -> None:
    ok = True
    for algoritmo in ALGORITMOS:
        calculado = calcular_ranking(algoritmo)
        taco = cargar_taco_ranking(algoritmo)
        merged = taco.merge(calculado, on="milestone", suffixes=("_taco", "_calc"))
        for variante in ("base", "surrogate"):
            diff = (merged[f"{variante}_taco"] - merged[f"{variante}_calc"]).abs().max()
            if diff > 5e-7:
                ok = False
            print(f"{algoritmo.upper()} {variante}: diff max = {diff:.8f}")
        print(merged.to_string(index=False))
        print()
    if not ok:
        raise SystemExit("La validación no reproduce los rankings de TACOLAB.")
    print("Validación correcta: el ranking calculado reproduce TACOLAB.")


if __name__ == "__main__":
    main()
