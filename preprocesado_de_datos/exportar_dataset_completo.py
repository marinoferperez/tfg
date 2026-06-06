from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocesado_de_datos.preprocesado import inferir_slug_experimento
from preprocesado_de_datos.utils.path_utils import (
    ALGORITMOS_MH,
    detectar_tareas_dataset_completo,
    normalizar_funcion,
)
from preprocesado_de_datos.utils.utils import concatenar_runs, guardar_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Concatena las runs originales de un bloque y guarda un dataset completo "
            "sin balancear en benchmarking/offline/dataset_preprocesado."
        )
    )
    parser.add_argument(
        "--experiment-dir",
        required=True,
        help="Raiz del experimento, carpeta fX o carpeta QAP que contiene metricas_runs.",
    )
    parser.add_argument(
        "--algoritmo",
        choices=[*ALGORITMOS_MH, "todos"],
        default="todos",
        help="Algoritmo a exportar. Default: todos.",
    )
    parser.add_argument(
        "--funcion",
        default=None,
        help="Funcion CEC concreta, por ejemplo f1. Si no se indica, exporta todas las disponibles.",
    )
    return parser.parse_args()


def stats_fitness(f_arr):
    f_arr = np.asarray(f_arr, dtype=float)
    return {
        "min": float(np.min(f_arr)),
        "max": float(np.max(f_arr)),
        "media": float(np.mean(f_arr)),
        "mediana": float(np.median(f_arr)),
        "desv_tipica": float(np.std(f_arr)),
        "percentiles": {
            "p1": float(np.percentile(f_arr, 1)),
            "p5": float(np.percentile(f_arr, 5)),
            "p25": float(np.percentile(f_arr, 25)),
            "p50": float(np.percentile(f_arr, 50)),
            "p75": float(np.percentile(f_arr, 75)),
            "p95": float(np.percentile(f_arr, 95)),
            "p99": float(np.percentile(f_arr, 99)),
        },
    }


def dist_seeds(s_arr):
    unique_s, counts_s = np.unique(np.asarray(s_arr, dtype=np.int32), return_counts=True)
    return {str(int(s)): int(c) for s, c in zip(unique_s, counts_s)}


def construir_metadata_dataset_completo(dataset, rutas_npz):
    fitness = np.asarray(dataset["fitness"], dtype=float)
    seeds = np.asarray(dataset["seed"], dtype=np.int32)
    n_total = int(len(fitness))
    experimento_id = inferir_slug_experimento(rutas_npz)

    return {
        "preprocesado": "dataset_completo_sin_balanceo",
        "criterio_seleccion": {
            "variable_balanceo": "fitness",
            "tipo_bins": None,
            "unidad_estratificacion": "seed",
            "estratificar_por_fase": False,
            "politica": "sin_balanceo_concatenacion_runs",
        },
        "n_runs_entrada": int(len(rutas_npz)),
        "experimento_id": str(experimento_id),
        "parametros": {
            "n_bins": None,
            "max_por_bin": None,
            "tipo_bins": None,
            "random_state": None,
            "estratificar_por_fase": False,
        },
        "muestras": {
            "original": n_total,
            "balanceado": n_total,
            "retencion_pct": 100.0,
        },
        "fitness": {
            "original": stats_fitness(fitness),
            "balanceado": stats_fitness(fitness),
            "edges": [],
        },
        "distribucion_bins": {
            "antes": [],
            "despues": [],
        },
        "distribucion_seeds": {
            "antes": dist_seeds(seeds),
            "despues": dist_seeds(seeds),
        },
        "n_seeds_unicas": int(len(np.unique(seeds))),
        "shapes": {k: list(np.asarray(v).shape) for k, v in dataset.items()},
        "claves": sorted(dataset.keys()),
        "origen_runs": [str(Path(p).resolve()) for p in rutas_npz],
    }


def exportar_dataset_completo(rutas_npz, ruta_salida):
    dataset = concatenar_runs(rutas_npz)
    metadata = construir_metadata_dataset_completo(dataset, rutas_npz)
    guardar_dataset(dataset, ruta_salida, metadata)
    return {
        "dataset_path": str(Path(ruta_salida).resolve()),
        "metadata_path": str(Path(ruta_salida).with_suffix(".metadata.json").resolve()),
        "n_muestras": int(len(dataset["fitness"])),
        "n_runs_entrada": int(len(rutas_npz)),
        "claves": sorted(dataset.keys()),
    }


def ejecutar_exportacion_dataset_completo(experiment_dir, *, algoritmo="todos", funcion=None):
    funcion_norm = normalizar_funcion(funcion)
    tareas = detectar_tareas_dataset_completo(
        experiment_dir,
        algoritmo=algoritmo,
        funcion=funcion_norm,
    )
    resultados = []
    for tarea in tareas:
        rutas_npz = [str(path) for path in detectar_inputs_tarea(tarea)]
        resultado = exportar_dataset_completo(rutas_npz, tarea["out"])
        resultado.update(
            {
                "algoritmo": str(tarea["algoritmo"]),
                "funcion": tarea["funcion"],
                "experiment_dir": str(Path(tarea["experiment_dir"]).resolve()),
            }
        )
        resultados.append(resultado)
    return resultados


def detectar_inputs_tarea(tarea):
    from preprocesado_de_datos.utils.path_utils import resolver_inputs_experimento

    return resolver_inputs_experimento(
        tarea["experiment_dir"],
        str(tarea["algoritmo"]),
        funcion=tarea["funcion"],
    )


def main():
    args = parse_args()
    resultados = ejecutar_exportacion_dataset_completo(
        args.experiment_dir,
        algoritmo=args.algoritmo,
        funcion=args.funcion,
    )
    for resultado in resultados:
        bloque = resultado["funcion"] if resultado["funcion"] else "qap"
        print(
            f"[{resultado['algoritmo'].upper()} {bloque}] "
            f"runs={resultado['n_runs_entrada']} muestras={resultado['n_muestras']} "
            f"dataset={resultado['dataset_path']}"
        )


if __name__ == "__main__":
    main()
