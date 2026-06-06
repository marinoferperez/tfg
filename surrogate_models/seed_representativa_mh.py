#!/usr/bin/env python3
"""
seed_representativa_mh.py
─────────────────────────
Para cada combinación (función, metaheurística), encuentra la seed más
representativa del conjunto de seeds utilizadas (generalmente 15)
basándose EXCLUSIVAMENTE en el fitness final de la metaheurística.

Criterio:
  Se calcula el fitness final mediano de la MH para las 15 seeds y se
  selecciona la seed cuyo fitness final sea más cercano a esta mediana.
  En caso de empate (común en funciones fáciles donde varias seeds
  alcanzan el óptimo), se escoge la semilla de menor índice.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def normalizar_umbral_reinicio(valor):
    txt = str(valor if valor is not None else "").strip().lower()
    if txt in {"", "none", "nan"}:
        return None

    umbral = float(txt)
    if abs(umbral - 0.05) <= 1e-9:
        return 0.05
    if abs(umbral - 0.10) <= 1e-9:
        return 0.10
    return umbral


def clasificar_variante_reinicio(umbral):
    if umbral is None or pd.isna(umbral):
        return "sin_reinicio"
    if abs(float(umbral) - 0.05) <= 1e-9:
        return "reinicio_005"
    if abs(float(umbral) - 0.10) <= 1e-9:
        return "reinicio_010"
    return f"reinicio_{str(umbral).replace('.', 'p')}"


def etiqueta_variante(clave):
    etiquetas = {
        "sin_reinicio": "Sin reinicio",
        "reinicio_005": "Reinicio 5%",
        "reinicio_010": "Reinicio 10%",
    }
    return etiquetas.get(str(clave), str(clave))


def main():
    parser = argparse.ArgumentParser(
        description="Encuentra la seed más representativa basándose en el fitness de la MH."
    )
    parser.add_argument(
        "--runs-csv",
        type=str,
        required=True,
        help="Ruta al archivo runs.csv global (ej. metaheuristica_resultados/runs.csv).",
    )
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=15,
        help="Número de seeds iniciales a considerar (default: 15).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Ruta opcional de salida JSON con los resultados.",
    )
    args = parser.parse_args()

    runs_path = Path(args.runs_csv)
    if not runs_path.exists():
        raise FileNotFoundError(f"No existe el archivo de runs: {runs_path}")

    df_runs = pd.read_csv(runs_path)
    
    # Filtrar solo las primeras N seeds
    df_runs = df_runs[df_runs["semilla"] <= args.max_seeds].copy()
    
    # Asegurarnos de tener las columnas necesarias
    req_cols = {"problema", "algoritmo", "cec_funcid", "semilla", "fitness"}
    if not req_cols.issubset(df_runs.columns):
        raise ValueError(f"Faltan columnas en runs.csv. Requeridas: {req_cols}")

    if "variante_reinicio" not in df_runs.columns:
        if "reinicio_elitista_umbral_diversidad" in df_runs.columns:
            df_runs["reinicio_elitista_umbral_diversidad"] = df_runs[
                "reinicio_elitista_umbral_diversidad"
            ].apply(normalizar_umbral_reinicio)
            df_runs["variante_reinicio"] = df_runs["reinicio_elitista_umbral_diversidad"].apply(
                clasificar_variante_reinicio
            )
        else:
            df_runs["reinicio_elitista_umbral_diversidad"] = None
            df_runs["variante_reinicio"] = "sin_reinicio"
    elif "reinicio_elitista_umbral_diversidad" in df_runs.columns:
        df_runs["reinicio_elitista_umbral_diversidad"] = df_runs[
            "reinicio_elitista_umbral_diversidad"
        ].apply(normalizar_umbral_reinicio)
    else:
        df_runs["reinicio_elitista_umbral_diversidad"] = None

    # Ordenar por semilla para que en caso de empate se prefiera la menor
    df_runs = df_runs.sort_values("semilla")

    resultados = []
    
    # Agrupar por problema, algoritmo, cec_funcid y variante de reinicio.
    groups = df_runs.groupby(["problema", "algoritmo", "cec_funcid", "variante_reinicio"], dropna=False)
    
    for (problema, algoritmo, funcid, variante), df_group in groups:
        df_group = df_group.copy()
        
        # 1. Mediana del fitness final
        median_fitness = df_group["fitness"].median()
        df_group["dist_fitness"] = (df_group["fitness"] - median_fitness).abs()
        
        # 2. Seleccionar la seed con mínima distancia
        # Como iteramos con idxmin sobre el dataframe ordenado, el primer índice (la menor semilla) en empatar gana.
        best_idx = df_group["dist_fitness"].idxmin()
        best_seed_row = df_group.loc[best_idx]
        
        best_seed = int(best_seed_row["semilla"])
        
        # Preparar resultados
        res = {
            "problema": problema,
            "algoritmo": algoritmo,
            "cec_funcid": int(funcid),
            "funcion": f"f{int(funcid)}",
            "variante_reinicio": str(variante),
            "reinicio_elitista_umbral_diversidad": (
                None
                if pd.isna(best_seed_row.get("reinicio_elitista_umbral_diversidad"))
                else best_seed_row.get("reinicio_elitista_umbral_diversidad")
            ),
            "seed_representativa": best_seed,
            "median_fitness": float(median_fitness),
            "seed_fitness": float(best_seed_row["fitness"]),
            "dist_fitness": float(best_seed_row["dist_fitness"]),
        }
        
        resultados.append(res)
        
        # Imprimir por pantalla
        print(f"[{res['funcion']} / {algoritmo.upper()} / {etiqueta_variante(variante)}]")
        print(f"  Mediana Fitness: {median_fitness:.4f}")
        print(f"  => Seed Representativa Escogida: {best_seed} (Fitness: {best_seed_row['fitness']:.4f}, Distancia: {best_seed_row['dist_fitness']:.4e})")
        print()

    # Guardar JSON
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(resultados, fh, indent=2, ensure_ascii=False, default=str)
        print(f"Resultados guardados en {out_path}")
    else:
        out_path = runs_path.parent / "seed_representativa_mh_unicriterio.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(resultados, fh, indent=2, ensure_ascii=False, default=str)
        print(f"Resultados guardados automáticamente en {out_path}")


if __name__ == "__main__":
    main()
