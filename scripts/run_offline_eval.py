"""
Ejecuta el benchmark offline de modelos subrogados sobre datasets por seed.

Punto de entrada unificado para las estrategias de evaluacion temporal:

  --strategy cumulative     Entrena con todos los batches acumulados hasta el
                            batch actual y valida en el bloque siguiente.
  --strategy non_cumulative Entrena solo con el batch actual (ventana fija) y
                            valida en el bloque siguiente.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.experiment_paths import ALGORITMOS_MH, normalizar_funcion
from src.utils.experiment_io import mostrar
from src.utils.benchmark.batches_eval_splitter import (
    construir_casos_acumulativos,
    construir_casos_no_acumulativos,
)
from src.utils.benchmark.benchmark_offline import ejecutar_benchmark_temporal, cargar_model_kwargs
from src.utils.benchmark.benchmark_io import (
    guardar_artefactos_batches,
    guardar_artefactos_modelo,
    imprimir_resumen,
)
from src.utils.benchmark.surrogate_paths import (
    resolver_inputs_benchmark,
    resolver_rutas_salida_benchmark,
)

STRATEGIES = {
    "cumulative": {
        "protocol": "acumulativo",
        "split_strategy": "temporal_acumulativo_futuro",
        "constructor_casos": construir_casos_acumulativos,
        "description": (
            "Benchmark temporal acumulativo: entrena con todos los batches "
            "anteriores al actual y valida en el bloque siguiente."
        ),
    },
    "non_cumulative": {
        "protocol": "no_acumulativo",
        "split_strategy": "temporal_no_acumulativo_futuro",
        "constructor_casos": construir_casos_no_acumulativos,
        "description": (
            "Benchmark temporal no acumulativo: entrena solo con el batch actual "
            "(ventana fija) y valida en el bloque siguiente."
        ),
    },
}


def parse_args():
    """Lee y devuelve los argumentos de linea de comandos."""
    parser = argparse.ArgumentParser(
        description="Benchmark offline de modelos subrogados sobre datasets CEC2017 por seed.",
    )
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGIES),
        required=True,
        help="Estrategia de evaluacion temporal: cumulative | non_cumulative.",
    )
    parser.add_argument(
        "--experiment-dir",
        required=True,
        help="Directorio raiz del experimento con los datasets por seed.",
    )
    parser.add_argument(
        "--benchmark-subdir",
        default="benchmarking",
        help=(
            "Subdirectorio relativo dentro del experimento donde guardar los resultados "
            "del benchmark. Por defecto: benchmarking."
        ),
    )
    parser.add_argument(
        "--algorithm",
        default="all",
        choices=[*ALGORITMOS_MH, "all"],
        help="Metaheuristica evaluada. 'all' ejecuta AGE, DE y SHADE.",
    )
    parser.add_argument(
        "--cec-funcid",
        required=True,
        help="Funcion CEC2017 a evaluar (ej. 1, f1, cec2017_f1).",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help="Lista opcional de datasets por seed. Si no se indica, se resuelven desde --experiment-dir.",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["rbf", "svr", "mlp", "rsm", "random_forest", "hgb", "lasso", "xgboost"],
        help="Modelo subrogado a evaluar.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semilla para reproducibilidad del modelo. Por defecto 42.",
    )
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=None,
        help="Numero maximo de seeds a usar. Si no se indica, se usan todas las disponibles.",
    )
    parser.add_argument(
        "--seed-selection-random-state",
        type=int,
        default=42,
        help=(
            "Semilla para seleccionar aleatoriamente el subconjunto de seeds cuando "
            "se usa --max-seeds. Por defecto 42."
        ),
    )
    parser.add_argument(
        "--convergence-truncation",
        action="store_true",
        default=False,
        help=(
            "Truncar el dataset en el ultimo batch con mejora real del running best. "
            "Los batches posteriores a la ultima mejora se descartan antes de construir los casos."
        ),
    )
    parser.add_argument(
        "--model-params-json",
        default=None,
        help="JSON con hiperparametros del modelo subrogado.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Ruta de salida para el JSON de metricas agregadas. Por defecto se genera automaticamente.",
    )
    parser.add_argument(
        "--runs-out",
        default=None,
        help="Ruta de salida para el CSV de metricas por run. Por defecto se genera automaticamente.",
    )
    parser.add_argument(
        "--runs-json-out",
        default=None,
        help="Ruta de salida para el JSON de metricas por run. Por defecto se genera automaticamente.",
    )
    parser.add_argument(
        "--save-runs-json",
        action="store_true",
        help="Generar tambien *_runs.json. Por defecto solo se guarda *_runs.csv.",
    )
    parser.add_argument(
        "--batch-runs",
        action="store_true",
        help=(
            "Generar tambien *_runs.csv por batch. Por defecto solo se guardan "
            "metricas por batch para poder construir resumenes."
        ),
    )
    parser.add_argument(
        "--errors-out",
        default=None,
        help="Ruta de salida para el CSV de errores por muestra. Si no se indica, no se genera.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Si se indica, muestra el bloque de configuracion antes de ejecutar.",
    )
    return parser.parse_args()


def main():
    """Punto de entrada del benchmark offline."""
    args = parse_args()
    cfg = STRATEGIES[args.strategy]
    protocol = cfg["protocol"]
    split_strategy = cfg["split_strategy"]
    constructor_casos = cfg["constructor_casos"]

    mostrar(args, "Configuracion offline:", flush=True)
    mostrar(args, f"  strategy={args.strategy}", flush=True)
    mostrar(args, f"  experiment_dir={args.experiment_dir}", flush=True)
    mostrar(args, f"  algorithm={args.algorithm}", flush=True)
    mostrar(args, f"  cec_funcid={args.cec_funcid}", flush=True)
    mostrar(args, f"  model={args.model}", flush=True)
    mostrar(args, f"  random_state={args.random_state}", flush=True)
    mostrar(args, f"  max_seeds={args.max_seeds}", flush=True)
    mostrar(args, f"  convergence_truncation={args.convergence_truncation}", flush=True)
    mostrar(args, f"  benchmark_subdir={args.benchmark_subdir}", flush=True)

    algoritmos = ALGORITMOS_MH if args.algorithm == "all" else (args.algorithm,)
    for algoritmo in algoritmos:
        args_algoritmo = argparse.Namespace(**vars(args))
        args_algoritmo.algorithm = algoritmo

        dataset_paths = resolver_inputs_benchmark(args_algoritmo)
        (
            _,
            model_dir,
            ruta_metricas,
            ruta_runs_csv,
            ruta_runs_json,
            ruta_errores,
        ) = resolver_rutas_salida_benchmark(args_algoritmo, dataset_paths, protocol)

        metricas = ejecutar_benchmark_temporal(
            dataset_paths=dataset_paths,
            funcion=normalizar_funcion(args_algoritmo.cec_funcid),
            algoritmo=args_algoritmo.algorithm,
            model_name=args_algoritmo.model,
            model_kwargs=cargar_model_kwargs(args_algoritmo.model_params_json),
            constructor_casos=constructor_casos,
            protocol=protocol,
            split_strategy=split_strategy,
            random_state=args_algoritmo.random_state,
            seed_selection_random_state=args_algoritmo.seed_selection_random_state,
            collect_sample_errors=(ruta_errores is not None),
            truncar_convergencia=args_algoritmo.convergence_truncation,
        )

        guardar_artefactos_modelo(
            ruta_metricas=ruta_metricas,
            ruta_runs_csv=ruta_runs_csv,
            ruta_runs_json=ruta_runs_json,
            ruta_errores=ruta_errores,
            metricas=metricas,
            guardar_runs=True,
        )
        guardar_artefactos_batches(
            model_dir=model_dir,
            model_name=args_algoritmo.model,
            ruta_errores_base=ruta_errores,
            metricas=metricas,
            guardar_runs=args_algoritmo.batch_runs,
        )

        imprimir_resumen(metricas)


if __name__ == "__main__":
    main()
