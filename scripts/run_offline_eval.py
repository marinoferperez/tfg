"""
Ejecuta el benchmark offline de modelos subrogados sobre datasets por seed.

Punto de entrada unificado para las estrategias de evaluacion temporal:

  --strategy cumulative     Entrena con todos los bloques acumulados hasta el
                            bloque actual y valida en el bloque siguiente.
  --strategy non_cumulative Entrena solo con el bloque actual (ventana fija) y
                            valida en el bloque siguiente.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.experiment_paths import gestiona_algoritmos, normalizar_funcion
from src.utils.experiment_io import mostrar
from src.utils.benchmark.blocks_eval_splitter import (
    construir_casos_acumulativos,
    construir_casos_no_acumulativos,
)
from src.utils.benchmark.benchmark_offline import ejecutar_benchmark_temporal, cargar_hiper_subrogado
from src.utils.benchmark.benchmark_io import (
    guardar_artefactos_bloques,
    guardar_artefactos_modelo,
    imprimir_resumen,
)
from src.utils.benchmark.surrogate_paths import (
    seleccionar_datasets,
    rutas_salida_benchmark,
)

STRATEGIES = {
    "cumulative": {
        "protocol": "acumulativo",
        "split_strategy": "temporal_acumulativo_futuro",
        "constructor_casos": construir_casos_acumulativos,
        "description": (
            "Benchmark temporal acumulativo: entrena con todos los bloques "
            "anteriores al actual y valida en el bloque siguiente."
        ),
    },
    "non_cumulative": {
        "protocol": "no_acumulativo",
        "split_strategy": "temporal_no_acumulativo_futuro",
        "constructor_casos": construir_casos_no_acumulativos,
        "description": (
            "Benchmark temporal no acumulativo: entrena solo con el bloque actual "
            "(ventana fija) y valida en el bloque siguiente."
        ),
    },
}


def expandir_funciones(funcion_arg):
    """Convierte el argumento --cec-funcid (lista nargs="+") en una tupla de nombres normalizados."""
    tokens = []
    for parte in funcion_arg:
        for tk in str(parte).split(","):
            tk = tk.strip().lower()
            if tk:
                tokens.append(tk)
    if "all" in tokens:
        return tuple(f"f{i}" for i in range(1, 31))
    return tuple(normalizar_funcion(tk) for tk in tokens)


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
        "--output-dir",
        default=None,
        help=(
            "Directorio raiz donde guardar los resultados del benchmark. "
            "Si no se indica, se genera automaticamente como results/offline_<nombre_experimento>."
        ),
    )
    parser.add_argument(
        "--benchmark-subdir",
        default="offline",
        help=(
            "Subdirectorio relativo dentro del experimento donde guardar los resultados "
            "del benchmark offline. Por defecto: offline."
        ),
    )
    parser.add_argument(
        "--algorithm",
        nargs="+",
        default=["all"],
        help="Metaheuristica a ejecutar. Acepta age, de, shade, all, listas separadas por espacios o comas. Por defecto all.",
    )
    parser.add_argument(
        "--cec-funcid",
        nargs="+",
        required=True,
        help="Funciones CEC2017 a evaluar. Acepta lista (1 4 10), CSV (1,4,10) o 'all'.",
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
        "--seed",
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
        "--selection-seed",
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
            "Truncar el dataset en el ultimo bloque con mejora real del running best. "
            "Los bloques posteriores a la ultima mejora se descartan antes de construir los casos."
        ),
    )
    parser.add_argument(
        "--surrogate-params-json",
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
        "--bloque-runs",
        action="store_true",
        help=(
            "Generar tambien *_runs.csv por bloque. Por defecto solo se guardan "
            "metricas por bloque para poder construir resumenes."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Si se indica, muestra informacion de progreso por terminal. Por defecto False.",
    )
    return parser.parse_args()


def main():
    """Punto de entrada del benchmark offline."""
    args = parse_args()
    if args.output_dir is None:
        nombre_exp = Path(args.experiment_dir).resolve().name
        args.output_dir = str(ROOT / "results" / f"offline_{nombre_exp}")
    cfg = STRATEGIES[args.strategy]
    protocol = cfg["protocol"]
    split_strategy = cfg["split_strategy"]
    constructor_casos = cfg["constructor_casos"]

    mostrar(args, "Configuracion offline:", flush=True)
    mostrar(args, f"  strategy={args.strategy}", flush=True)
    mostrar(args, f"  experiment_dir={args.experiment_dir}", flush=True)
    mostrar(args, f"  output_dir={args.output_dir}", flush=True)
    mostrar(args, f"  algorithm={args.algorithm}", flush=True)
    mostrar(args, f"  cec_funcid={args.cec_funcid}", flush=True)
    mostrar(args, f"  model={args.model}", flush=True)
    mostrar(args, f"  seed={args.seed}", flush=True)
    mostrar(args, f"  max_seeds={args.max_seeds}", flush=True)
    mostrar(args, f"  convergence_truncation={args.convergence_truncation}", flush=True)
    mostrar(args, f"  benchmark_subdir={args.benchmark_subdir}", flush=True)

    funciones = expandir_funciones(args.cec_funcid)
    algoritmos = gestiona_algoritmos(args.algorithm)
    for funcion in funciones:
        for algoritmo in algoritmos:
            args_algoritmo = argparse.Namespace(**vars(args))
            args_algoritmo.cec_funcid = funcion
            args_algoritmo.algorithm = algoritmo

            dataset_paths = seleccionar_datasets(args_algoritmo)
            (
                _,
                model_dir,
                ruta_metricas,
                ruta_runs_csv,
                ruta_runs_json,
            ) = rutas_salida_benchmark(args_algoritmo, dataset_paths, protocol)

            metricas = ejecutar_benchmark_temporal(
                dataset_paths=dataset_paths,
                funcion=normalizar_funcion(args_algoritmo.cec_funcid),
                algoritmo=args_algoritmo.algorithm,
                nombre_subrogado=args_algoritmo.model,
                hiper_subrogado=cargar_hiper_subrogado(args_algoritmo.surrogate_params_json),
                constructor_casos=constructor_casos,
                protocolo=protocol,
                estrategia_split=split_strategy,
                random_state=args_algoritmo.seed,
                seed_selection_random_state=args_algoritmo.selection_seed,
                truncar_convergencia=args_algoritmo.convergence_truncation,
            )

            guardar_artefactos_modelo(
                ruta_metricas=ruta_metricas,
                ruta_runs_csv=ruta_runs_csv,
                ruta_runs_json=ruta_runs_json,
                metricas=metricas,
                guardar_runs=True,
            )
            guardar_artefactos_bloques(
                dir_subrogado=model_dir,
                nombre_subrogado=args_algoritmo.model,
                metricas=metricas,
                guardar_runs=args_algoritmo.bloque_runs,
            )

            imprimir_resumen(metricas)


if __name__ == "__main__":
    main()
