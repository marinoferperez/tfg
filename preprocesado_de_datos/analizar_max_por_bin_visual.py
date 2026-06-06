from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocesado_de_datos.utils.path_utils import (
    detectar_tareas_seleccion_bins,
    escribir_json,
    normalizar_funcion,
    resolver_inputs_experimento,
)

FASES = (
    (0.0, 0.2, "0-20"),
    (0.2, 0.4, "20-40"),
    (0.4, 0.6, "40-60"),
    (0.6, 0.8, "60-80"),
    (0.8, 1.0, "80-100"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analiza visualmente un experimento a partir de los datasets por run y propone una orientacion para max_por_bin."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help="Lista de rutas a dataset_*.npz de un mismo algoritmo/problema.",
    )
    parser.add_argument(
        "--experiment-dir",
        default=None,
        help=(
            "Directorio de experimento. En CEC2017 puede ser la raiz que contiene las carpetas f* "
            "o una carpeta concreta fX. En QAP debe ser una carpeta de experimento que contenga metricas_runs."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Ruta opcional para guardar el resumen en JSON.",
    )
    parser.add_argument(
        "--algoritmo",
        default="todos",
        choices=["age", "de", "todos"],
        help=(
            "Algoritmo analizado. Con --experiment-dir, 'todos' genera age y de automaticamente "
            "en la ruta estandar de benchmarking/offline/dataset_preprocesado/seleccion_muestras."
        ),
    )
    parser.add_argument(
        "--funcion",
        default=None,
        help=(
            "Funcion CEC a analizar cuando se usa --experiment-dir. "
            "Acepta formatos como 'f1' o '1'. Si no se indica, se incluyen todas las funciones."
        ),
    )
    return parser.parse_args()


# resolver_inputs identifica que archivos deben procesarse.

def resolver_inputs(inputs, experiment_dir, algoritmo, funcion=None):
    if inputs:
        return inputs

    if not experiment_dir:
        raise ValueError("Debes indicar --inputs o --experiment-dir.")
    if algoritmo not in {"age", "de"}:
        raise ValueError("resolver_inputs solo acepta un algoritmo concreto: age o de.")
    rutas = resolver_inputs_experimento(experiment_dir, algoritmo, funcion=funcion)
    return [str(r) for r in rutas]


# asignar_fases_relativas divide la historia temporal de cada ejecución independiente (por seed) en 5 tramos 

def asignar_fases_relativas(eval_ids: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    fases = np.full(eval_ids.shape, -1, dtype=np.int32)
    for seed in np.unique(seeds):
        mascara_seed = seeds == seed
        evals_seed = eval_ids[mascara_seed].astype(float)
        if evals_seed.size == 0:
            continue
        total = max(float(np.max(evals_seed)), 1.0)
        progreso = np.clip(evals_seed / total, 0.0, 1.0)
        fases_seed = np.full(evals_seed.shape, -1, dtype=np.int32)
        for idx, (inicio, fin, _etq) in enumerate(FASES):
            if idx < len(FASES) - 1:
                mascara_fase = (progreso >= inicio) & (progreso < fin)
            else:
                mascara_fase = (progreso >= inicio) & (progreso <= fin)
            fases_seed[mascara_fase] = idx
        fases[mascara_seed] = fases_seed
    if np.any(fases < 0):
        raise ValueError("No se pudo asignar fase a todas las muestras.")
    return fases

# cargar_experimento carga los distintos datasets

def cargar_experimento(rutas_npz: list[str]):
    fitness_runs = []
    evals_runs = []
    seeds_runs = []
    for ruta in sorted(Path(p) for p in rutas_npz):
        with np.load(ruta, allow_pickle=True) as data:
            if "fitness" not in data or "eval_id" not in data:
                raise ValueError(f"{ruta} no contiene 'fitness' y 'eval_id'.")
            fitness = np.asarray(data["fitness"], dtype=float).reshape(-1)
            eval_id = np.asarray(data["eval_id"], dtype=int).reshape(-1)
        if fitness.size != eval_id.size:
            raise ValueError(f"{ruta} tiene longitudes inconsistentes entre fitness y eval_id.")
        mascara = np.isfinite(fitness)
        fitness = fitness[mascara]
        eval_id = eval_id[mascara]
        seed = int(ruta.parent.name.rsplit("_s", 1)[1])
        fitness_runs.append(fitness)
        evals_runs.append(eval_id)
        seeds_runs.append(np.full(fitness.shape, seed, dtype=np.int32))
    fitness = np.concatenate(fitness_runs)
    evals = np.concatenate(evals_runs)
    seeds = np.concatenate(seeds_runs)
    fases = asignar_fases_relativas(evals, seeds)
    return fitness, evals, seeds, fases


def hist_counts(valores: np.ndarray, edges: np.ndarray):
    counts, _ = np.histogram(valores, bins=edges)
    total = max(int(counts.sum()), 1)
    return counts, counts / total

# analizar extrae las distintas conclusiones.
# define un rango visual seguro para evitar outliers extremos limitando el espacio de analisis desde el p0.5 hasta p99.5 de los fitness.
# construye un hist de 60 barras y calcula distintas metricas globales como "el ancho de la distribucion", "que cuota tiene el bin mas lleno"...

# este procedimiento se repite fase por fase para obtener como evoluciona la dispersion estadistica fase a fase

def analizar(rutas_npz: list[str], algoritmo: str):
    fitness, evals, seeds, fases = cargar_experimento(rutas_npz)

    p005, p995 = np.percentile(fitness, [0.5, 99.5])
    vis = fitness[(fitness >= p005) & (fitness <= p995)]
    n_bins_hist = 60
    edges = np.linspace(p005, p995, n_bins_hist + 1)
    counts_global, shares_global = hist_counts(vis, edges)

    global_stats = {
        "visual_range_p0.5_p99.5": [float(p005), float(p995)],
        "max_bin_share": float(np.max(shares_global)),
        "top3_bins_share": float(np.sort(shares_global)[-3:].sum()),
        "occupied_bins_ratio": float(np.mean(counts_global > 0)),
        "width_p95_p5": float(np.percentile(fitness, 95) - np.percentile(fitness, 5)),
        "tail_low_p5_share": float(np.mean(fitness < np.percentile(fitness, 5))),
        "tail_high_p95_share": float(np.mean(fitness > np.percentile(fitness, 95))),
    }

    phase_stats = []
    for idx, (_inicio, _fin, etiqueta) in enumerate(FASES):
        arr = fitness[fases == idx]
        arr_vis = arr[(arr >= p005) & (arr <= p995)]
        counts_fase, shares_fase = hist_counts(arr_vis, edges)
        phase_stats.append(
            {
                "phase": etiqueta,
                "n": int(arr.size),
                "width_p95_p5": float(np.percentile(arr, 95) - np.percentile(arr, 5)),
                "max_bin_share": float(np.max(shares_fase)) if arr_vis.size > 0 else 0.0,
                "occupied_bins_ratio": float(np.mean(counts_fase > 0)) if arr_vis.size > 0 else 0.0,
            }
        )

    width_ratio = phase_stats[-1]["width_p95_p5"] / max(phase_stats[0]["width_p95_p5"], 1e-12)
    peak_ratio = phase_stats[-1]["max_bin_share"] / max(phase_stats[0]["max_bin_share"], 1e-12)

    booleans = {
        "pico_global_alto": global_stats["max_bin_share"] > 0.08,
        "concentracion_global_fuerte_top3": global_stats["top3_bins_share"] > 0.20,
        "fases_finales_colapsadas": width_ratio < 0.35,
        "pico_tardio_mas_agudo": peak_ratio > 1.6,
        "distribucion_global_extensa": global_stats["occupied_bins_ratio"] > 0.65,
    }

    if booleans["pico_global_alto"] and booleans["fases_finales_colapsadas"]:
        recomendacion = "bajar mucho desde 3000"
        sugerido = 1000
        comentario = (
            "La masa del experimento esta fuertemente concentrada y las fases finales colapsan sobre una zona estrecha. "
            "Mantener 3000 probablemente conserva demasiada redundancia. Tiene sentido probar un recorte agresivo."
        )
    elif booleans["concentracion_global_fuerte_top3"] or (
        booleans["fases_finales_colapsadas"] and booleans["pico_tardio_mas_agudo"]
    ):
        recomendacion = "bajar moderadamente desde 3000"
        sugerido = 1500
        comentario = (
            "Hay señales claras de concentracion o de colapso progresivo en fases finales, pero no un dominio tan extremo de la masa global. "
            "Conviene recortar, aunque no de forma tan agresiva."
        )
    else:
        recomendacion = "mantener cerca de 3000"
        sugerido = 3000
        comentario = (
            "La distribucion global no esta dominada por unos pocos picos y la estructura por fases no sugiere una sobresaturacion extrema. "
            "No hay evidencia fuerte para bajar de forma agresiva el max_por_bin."
        )

    return {
        "algoritmo": algoritmo,
        "n_runs": int(len(np.unique(seeds))),
        "n_samples": int(fitness.size),
        "global": global_stats,
        "phases": phase_stats,
        "comparativa_fases": {
            "width_ratio_80_100_vs_0_20": float(width_ratio),
            "peak_ratio_80_100_vs_0_20": float(peak_ratio),
        },
        "booleans": booleans,
        "recomendacion_visual": recomendacion,
        "max_por_bin_sugerido": int(sugerido),
        "comentario_salida": comentario,
    }


def main():
    args = parse_args()
    if args.inputs or args.out or args.algoritmo in {"age", "de"}:
        if args.algoritmo == "todos":
            raise ValueError(
                "Con el modo manual (--inputs o --out) debes indicar un algoritmo concreto con --algoritmo age|de."
            )
        rutas_npz = resolver_inputs(args.inputs, args.experiment_dir, args.algoritmo, args.funcion)
        resumen = analizar(rutas_npz, args.algoritmo)
        resumen["n_inputs"] = int(len(rutas_npz))
        resumen["source"] = {
            "experiment_dir": str(args.experiment_dir) if args.experiment_dir else None,
            "funcion": normalizar_funcion(args.funcion),
            "inputs_resueltos": rutas_npz,
        }
        if args.out:
            escribir_json(args.out, resumen)
        print(json.dumps(resumen, ensure_ascii=False, indent=2))
        return

    tareas = detectar_tareas_seleccion_bins(
        args.experiment_dir,
        algoritmo=args.algoritmo,
        funcion=args.funcion,
    )
    resumen_global = []
    for tarea in tareas:
        rutas_npz = resolver_inputs(
            None, tarea["experiment_dir"], tarea["algoritmo"], tarea["funcion"]
        )
        resumen = analizar(rutas_npz, tarea["algoritmo"])
        resumen["n_inputs"] = int(len(rutas_npz))
        resumen["source"] = {
            "experiment_dir": tarea["experiment_dir"],
            "funcion": tarea["funcion"],
            "inputs_resueltos": rutas_npz,
        }
        escribir_json(tarea["out"], resumen)
        resumen_global.append(
            {
                "algoritmo": tarea["algoritmo"],
                "funcion": tarea["funcion"],
                "out": tarea["out"],
                "max_por_bin_sugerido": resumen["max_por_bin_sugerido"],
                "recomendacion_visual": resumen["recomendacion_visual"],
            }
        )
    print(json.dumps({"n_tareas": len(resumen_global), "resultados": resumen_global}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
