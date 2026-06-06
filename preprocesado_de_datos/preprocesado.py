"""
preprocesado.py – Script principal para el balanceo a la baja de los datasets.

Orquesta las funciones de utils.py y fitness_utils.py para:
1. Concatenar todas las runs (.npz) de un experimento.
2. Dividir el rango de fitness en intervalos.
3. Submuestrear cada intervalo estratificando por seed.
4. Guardar el dataset resultante y un resumen en .metadata.json.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocesado_de_datos.utils.utils import (
    concatenar_runs,
    filtrar_dataset,
    guardar_dataset,
)
from preprocesado_de_datos.utils.path_utils import (
    asegurar_directorio,
    escribir_csv_dicts,
    escribir_json,
)
from preprocesado_de_datos.utils.fitness_utils import (
    balanceo_a_la_baja,
    contabilizar_muestras_por_bin,
    asignar_bins_fitness,
    construir_fases_relativas_por_seed,
)


def inferir_slug_experimento(rutas_npz):
    rutas = [Path(r) for r in rutas_npz]
    if len(rutas) == 0:
        return "experimento_desconocido"

    primera = rutas[0]
    partes = primera.parts

    nombre_experimento = None
    for parte in partes:
        if "experimentos" in str(parte):
            nombre_experimento = parte
            break
    if nombre_experimento is None:
        for idx, parte in enumerate(partes):
            if parte == "metricas_runs" and idx >= 1:
                nombre_experimento = partes[idx - 1]
                break
    if nombre_experimento is None:
        nombre_experimento = primera.parent.name

    run_dir = primera.parent.name
    run_tokens = run_dir.split("_s", 1)[0]
    match_tam = re.search(r"(?:^|_)tam_(\d+)(?:_|$)", nombre_experimento)
    if match_tam is not None:
        return f"experimentos_tam_{match_tam.group(1)}_{run_tokens}"
    return f"experimentos_{run_tokens}"


def resolver_ruta_salida(rutas_npz, ruta_salida):
    if ruta_salida is not None:
        return Path(ruta_salida)

    slug = inferir_slug_experimento(rutas_npz)
    return Path("preprocesado_de_datos/comparacion_balanceo") / slug / "dataset_balanceado.npz"


def construir_candidatos_max_por_bin(valores):
    if isinstance(valores, int):
        return [int(valores)]

    valores = [int(v) for v in valores]
    if len(valores) == 1:
        return valores
    if len(valores) != 2:
        raise ValueError("--max-por-bin debe recibir uno o dos valores.")

    minimo, maximo = valores
    if minimo <= 0 or maximo <= 0:
        raise ValueError("Los valores de --max-por-bin deben ser positivos.")
    if minimo > maximo:
        raise ValueError("Si indicas un rango, el primer valor debe ser <= que el segundo.")
    return list(range(minimo, maximo + 1, 500))


def resolver_ruta_salida_barrido(ruta_base, max_por_bin):
    ruta_base = Path(ruta_base)
    if ruta_base.suffix == ".npz":
        stem = ruta_base.stem
        return ruta_base.with_name(f"{stem}_max_{max_por_bin}.npz")
    return ruta_base / f"dataset_balanceado_max_{max_por_bin}.npz"

def procesar_experimento(
    rutas_npz,
    ruta_salida,
    n_bins,
    max_por_bin,
    tipo_bins,
    rng_seed,
    estratificar_por_fase=False,
):
    """
    Carga los datasets, aplica el balanceo y guarda el resultado junto con
    la metadata del proceso.
    """
    ruta_salida = resolver_ruta_salida(rutas_npz, ruta_salida)
    slug_experimento = inferir_slug_experimento(rutas_npz)

    print(f"Cargando y concatenando {len(rutas_npz)} runs...")
    dataset_completo = concatenar_runs(rutas_npz)

    fitness = dataset_completo["fitness"]
    seeds = dataset_completo["seed"]
    fases = (
        construir_fases_relativas_por_seed(dataset_completo["eval_id"], dataset_completo["seed"])
        if estratificar_por_fase
        else None
    )
    
    n_original = len(fitness)
    print(f"Dataset original: {n_original} muestras.")

    # conteo previo para la metadata
    bin_ids_pre, edges = asignar_bins_fitness(fitness, n_bins, tipo_bins=tipo_bins)
    conteo_pre = contabilizar_muestras_por_bin(bin_ids_pre, n_bins)

    print(f"Balanceando (n_bins={n_bins}, max_por_bin={max_por_bin}, tipo_bins={tipo_bins})...")
    indices_seleccionados, bin_ids_post, _ = balanceo_a_la_baja(
        fitness,
        seeds,
        n_bins=n_bins,
        max_por_bin=max_por_bin,
        tipo_bins=tipo_bins,
        random_state=rng_seed,
        fase_arr=fases,
        estratificar_por_fase=estratificar_por_fase,
    )

    n_balanceado = len(indices_seleccionados)
    print(f"Dataset balanceado: {n_balanceado} muestras.")

    # conteo posterior
    conteo_post = contabilizar_muestras_por_bin(bin_ids_post[indices_seleccionados], n_bins)

    # metadatos para analisis futuro (comprobar que no hay sesgo)

    # 1. funciones auxiliares para estadisticas y conteos
    def stats_fitness(f_arr):
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
            }
        }

    def dist_seeds(s_arr):
        unique_s, counts_s = np.unique(s_arr, return_counts=True)
        return {str(int(s)): int(c) for s, c in zip(unique_s, counts_s)}

    print("Generando dataset final...")
    dataset_final = filtrar_dataset(dataset_completo, indices_seleccionados)

    metadata = {
        "preprocesado": "balanceo_a_la_baja_estratificado",
        "criterio_seleccion": {
            "variable_balanceo": "fitness",
            "tipo_bins": tipo_bins,
            "unidad_estratificacion": "seed",
            "estratificar_por_fase": bool(estratificar_por_fase),
            "politica": "submuestreo_a_la_baja"
        },
        "n_runs_entrada": len(rutas_npz),
        "experimento_id": slug_experimento,
        "parametros": {
            "n_bins": n_bins,
            "max_por_bin": max_por_bin,
            "tipo_bins": tipo_bins,
            "random_state": rng_seed,
            "estratificar_por_fase": bool(estratificar_por_fase),
        },
        "muestras": {
            "original": int(n_original),
            "balanceado": int(n_balanceado),
            "retencion_pct": round((n_balanceado / n_original) * 100, 4) if n_original > 0 else 0
        },
        "fitness": {
            "original": stats_fitness(fitness),
            "balanceado": stats_fitness(fitness[indices_seleccionados]),
            "edges": edges.tolist()
        },
        "distribucion_bins": {
            "antes": conteo_pre.tolist(),
            "despues": conteo_post.tolist()
        },
        "distribucion_seeds": {
            "antes": dist_seeds(seeds),
            "despues": dist_seeds(seeds[indices_seleccionados])
        },
        "n_seeds_unicas": int(len(np.unique(seeds[indices_seleccionados]))),
        "shapes": {k: list(v.shape) for k, v in dataset_final.items()},
        "claves": sorted(dataset_completo.keys())
    }
    
    guardar_dataset(dataset_final, ruta_salida, metadata)
    print(f"Guardado exitosamente en: {ruta_salida}")
    return {
        "dataset_path": str(ruta_salida),
        "metadata_path": str(ruta_salida.with_suffix(".metadata.json")),
        "experimento_id": slug_experimento,
        "max_por_bin": int(max_por_bin),
        "n_original": int(n_original),
        "n_balanceado": int(n_balanceado),
        "retencion_pct": round((n_balanceado / n_original) * 100, 4) if n_original > 0 else 0.0,
        "fitness_original_min": float(np.min(fitness)),
        "fitness_original_max": float(np.max(fitness)),
        "fitness_balanceado_min": float(np.min(fitness[indices_seleccionados])),
        "fitness_balanceado_max": float(np.max(fitness[indices_seleccionados])),
    }


def guardar_resumen_barrido(ruta_base, resultados):
    ruta_base = Path(ruta_base)
    if ruta_base.suffix == ".npz":
        directorio = ruta_base.parent
    else:
        directorio = ruta_base
    directorio = asegurar_directorio(directorio)

    ruta_json = directorio / "resumen_barrido_max_por_bin.json"
    ruta_csv = directorio / "resumen_barrido_max_por_bin.csv"

    resumen = {
        "candidatos_max_por_bin": [int(r["max_por_bin"]) for r in resultados],
        "resultados": resultados,
        "comentario_salida": (
            "El barrido recorre max_por_bin en pasos de 500. "
            "Compara retencion, tamaño final y cobertura de fitness para decidir el valor mas adecuado."
        ),
    }
    escribir_json(ruta_json, resumen)

    campos = [
        "max_por_bin",
        "dataset_path",
        "metadata_path",
        "n_original",
        "n_balanceado",
        "retencion_pct",
        "fitness_original_min",
        "fitness_original_max",
        "fitness_balanceado_min",
        "fitness_balanceado_max",
    ]
    escribir_csv_dicts(ruta_csv, resultados, fieldnames=campos)

    print(f"Resumen del barrido guardado en: {ruta_json}")
    print(f"CSV del barrido guardado en: {ruta_csv}")
    print(json.dumps(resumen, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Balanceo a la baja estratificado por seed para datasets de metaheurísticas."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Lista de rutas a los archivos dataset_*.npz a concatenar y balancear.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Ruta donde guardar el .npz resultante. Si no se indica, se crea dentro de preprocesado_de_datos/comparacion_balanceo/<experimento_id>/dataset_balanceado.npz",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Número de intervalos de fitness. Default: 10.",
    )
    parser.add_argument(
        "--max-por-bin",
        type=int,
        nargs="+",
        default=[3000],
        help=(
            "Tope máximo de muestras a conservar por bin. "
            "Si indicas un valor, se ejecuta una sola vez. "
            "Si indicas dos valores, se interpreta como rango cerrado y se barre de 500 en 500. "
            "Ejemplo: --max-por-bin 1500 3000"
        ),
    )
    parser.add_argument(
        "--tipo-bins",
        choices=["uniformes", "cuantiles"],
        default="uniformes",
        help="Estrategia para delimitar los bins. 'cuantiles' ajusta los limites segun densidad de datos. Default: uniformes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para reproducibilidad del submuestreo aleatorio. Default: 42.",
    )
    parser.add_argument(
        "--estratificar-por-fase",
        action="store_true",
        help="Si se indica, dentro de cada bin de fitness reparte primero por fase relativa de ejecucion y despues por seed.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    candidatos = construir_candidatos_max_por_bin(args.max_por_bin)
    if len(candidatos) == 1:
        procesar_experimento(
            rutas_npz=args.inputs,
            ruta_salida=args.out,
            n_bins=args.n_bins,
            max_por_bin=candidatos[0],
            tipo_bins=args.tipo_bins,
            rng_seed=args.seed,
            estratificar_por_fase=bool(args.estratificar_por_fase),
        )
    else:
        ruta_base = resolver_ruta_salida(args.inputs, args.out)
        resultados = []
        for max_por_bin in candidatos:
            resultado = procesar_experimento(
                rutas_npz=args.inputs,
                ruta_salida=resolver_ruta_salida_barrido(ruta_base, max_por_bin),
                n_bins=args.n_bins,
                max_por_bin=max_por_bin,
                tipo_bins=args.tipo_bins,
                rng_seed=args.seed,
                estratificar_por_fase=bool(args.estratificar_por_fase),
            )
            resultados.append(resultado)
        guardar_resumen_barrido(ruta_base, resultados)
