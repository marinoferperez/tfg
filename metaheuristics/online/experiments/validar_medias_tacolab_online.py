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

FUNCIONES = range(1, 31)
MILESTONE = 100
TOL_TIE = 1e-12
TOL_MEDIA = 5e-5


def cargar_resultado(root_name: str, prefix: str, funcid: int) -> float:
    func_dir = RESULTS / root_name / f"f{funcid}"
    files = list(func_dir.glob(f"results_{prefix}_*/results_{funcid}_10.txt"))
    if len(files) != 1:
        raise RuntimeError(f"No se pudo resolver un único fichero para {root_name} f{funcid}: {files}")

    df = pd.read_csv(files[0])
    row = df[df["milestone"] == MILESTONE]
    if row.empty:
        raise RuntimeError(f"No existe milestone {MILESTONE} en {files[0]}")
    return float(row["error"].mean())


def calcular_medias(algoritmo: str) -> pd.DataFrame:
    prefix, base_root, surrogate_root = ALGORITMOS[algoritmo]
    rows = []
    for funcid in FUNCIONES:
        rows.append(
            {
                "funcid": funcid,
                "base": cargar_resultado(base_root, prefix, funcid),
                "surrogate": cargar_resultado(surrogate_root, prefix, funcid),
            }
        )
    return pd.DataFrame(rows).set_index("funcid")


def calcular_best(medias: pd.DataFrame) -> dict[str, int]:
    best = {"base": 0, "surrogate": 0}
    for _, row in medias.iterrows():
        valores = row[["base", "surrogate"]].astype(float)
        minimo = valores.min()
        for variante, valor in valores.items():
            if abs(valor - minimo) <= TOL_TIE:
                best[variante] += 1
    return best


def cargar_taco_media(algoritmo: str) -> tuple[pd.DataFrame, dict[str, int]]:
    text = (TACO / f"media_{algoritmo}_online.tex").read_text(encoding="utf-8")
    rows = []
    best = None

    for line in text.splitlines():
        stripped = line.strip()
        match = re.match(r"F(\d+)\s*&\s*([^&]+?)\s*&\s*([^\\]+?)\s*\\\\", stripped)
        if match:
            rows.append(
                {
                    "funcid": int(match.group(1)),
                    "base": float(match.group(2)),
                    "surrogate": float(match.group(3)),
                }
            )
            continue

        match_best = re.match(r"Best\s*&\s*(\d+)\s*&\s*(\d+)\s*\\\\", stripped)
        if match_best:
            best = {"base": int(match_best.group(1)), "surrogate": int(match_best.group(2))}

    if len(rows) != 30 or best is None:
        raise RuntimeError(f"No se pudo leer correctamente media_{algoritmo}_online.tex")
    return pd.DataFrame(rows).set_index("funcid"), best


def comparar_medias(taco: pd.DataFrame, calculado: pd.DataFrame) -> float:
    max_diff_rel = 0.0
    for funcid in FUNCIONES:
        for variante in ("base", "surrogate"):
            esperado = float(taco.loc[funcid, variante])
            observado = float(calculado.loc[funcid, variante])
            diff = abs(esperado - observado)
            escala = max(1.0, abs(esperado), abs(observado))
            diff_rel = diff / escala
            max_diff_rel = max(max_diff_rel, diff_rel)
            if diff_rel > TOL_MEDIA and diff > TOL_TIE:
                raise RuntimeError(
                    f"Diferencia en {funcid=} {variante=} "
                    f"TACOLAB={esperado:.8e} calculado={observado:.8e}"
                )
    return max_diff_rel


def main() -> None:
    for algoritmo in ALGORITMOS:
        calculado = calcular_medias(algoritmo)
        taco, best_taco = cargar_taco_media(algoritmo)
        diff = comparar_medias(taco, calculado)

        best_calc = calcular_best(calculado)
        if best_calc != best_taco:
            raise RuntimeError(f"Best distinto para {algoritmo}: TACOLAB={best_taco}, calculado={best_calc}")

        print(
            f"{algoritmo.upper()}: medias OK (diff rel max {diff:.8f}), "
            f"Best={best_calc}"
        )

    print("Validación correcta: las medias por función reproducen TACOLAB.")


if __name__ == "__main__":
    main()
