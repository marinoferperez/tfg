from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

# Añadir root al path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocesado_de_datos.utils.utils import filtrar_dataset, guardar_dataset


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


def contar_bins_con_edges(fitness, edges):
    edges = np.asarray(edges, dtype=float)
    if edges.size < 2:
        return [int(len(fitness))]
    bin_ids = np.digitize(np.asarray(fitness, dtype=float), edges[1:-1], right=False).astype(np.int32)
    conteos = np.bincount(bin_ids, minlength=max(int(edges.size - 1), 1))
    return [int(v) for v in conteos.tolist()]

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Alinea múltiples datasets preprocesados al tamaño mínimo entre ellos "
            "solo cuando todos provienen de bins por cuantiles."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Rutas a los archivos dataset_balanceado.npz u homólogos correspondientes a los algoritmos a alinear."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para el muestreo aleatorio en la reducción de los datasets más grandes."
    )
    return parser.parse_args()


def procesar_alineamiento():
    args = parse_args()
    rutas_npz = [Path(p).resolve() for p in args.datasets]
    rng = np.random.default_rng(args.seed)

    datasets = []
    tipos_bins = []

    # 1. Cargar y revisar datos
    for path_npz in rutas_npz:
        if not path_npz.exists():
            raise FileNotFoundError(f"No se encontró: {path_npz}")
        
        path_meta = path_npz.with_suffix(".metadata.json")
        if not path_meta.exists():
            raise FileNotFoundError(f"Falta archivo de metadatos para: {path_npz}")

        with open(path_meta, "r") as f:
            meta = json.load(f)

        tipo_bins = meta.get("criterio_seleccion", {}).get("tipo_bins", "")
        tipos_bins.append(tipo_bins)

        ds = np.load(path_npz)
        datasets.append((path_npz, path_meta, meta, dict(ds)))
        
        print(f"Cargado {path_npz.name} -> {len(ds['fitness'])} muestras (Tipo Bins: {tipo_bins})")

    usan_cuantiles_todos = bool(tipos_bins) and all(tipo == "cuantiles" for tipo in tipos_bins)
    if not usan_cuantiles_todos:
        print(
            "\nNo todos los datasets usan 'cuantiles'. "
            "La alineacion se omite porque esta regla solo aplica cuando ambos algoritmos "
            "han quedado balanceados con el mismo criterio por cuantiles."
        )
        return

    # 2. Encontrar el tamaño mínimo
    tamanos = [len(ds_dict["fitness"]) for _, _, _, ds_dict in datasets]
    tam_minimo = min(tamanos)
    print(f"\nTamaño mínimo detectado: {tam_minimo} muestras.")

    # 3. Alinear y Sobrescribir (o guardar en subcarpeta)
    for path_npz, path_meta, meta, ds_dict in datasets:
        n_actual = len(ds_dict["fitness"])
        
        if n_actual == tam_minimo:
            print(f"[{path_npz.name}] Ya tiene el tamaño mínimo ({n_actual}). Se deja intacto.")
            meta["muestras"]["balanceado_pre_alineacion"] = n_actual
            meta["muestras"]["alineado"] = tam_minimo
            meta["muestras"]["retencion_pct_alineado"] = meta["muestras"].get("retencion_pct")
            meta["alineamiento_global"] = {
                "aplicado": True,
                "criterio_activacion": "todos_los_datasets_con_bins_cuantiles",
                "tipos_bins_detectados": list(tipos_bins),
                "n_original": n_actual,
                "n_alineado": tam_minimo,
                "comentario": (
                    "Este dataset ya era el mas pequeno. No requiere recortes adicionales "
                    "para igualar el tamaño comun entre algoritmos."
                ),
            }
            with open(path_meta, "w") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            continue

        print(f"[{path_npz.name}] Recortando de {n_actual} a {tam_minimo} muestras de forma aleatoria estructurada.")
        
        # El dataset ya fue equilibrado por cuantiles. Un recorte aleatorio adicional
        # sobre el conjunto mayor no altera materialmente esa representatividad global.
        indices_sel = np.sort(rng.choice(n_actual, size=tam_minimo, replace=False))
        ds_recortado = filtrar_dataset(ds_dict, indices_sel)
        
        # Actualizamos la shape metadata
        meta["shapes"] = {k: list(np.asarray(v).shape) for k, v in ds_recortado.items()}
        meta["fitness"]["balanceado"] = stats_fitness(ds_recortado["fitness"])
        meta["distribucion_seeds"]["despues"] = dist_seeds(ds_recortado["seed"])
        meta["n_seeds_unicas"] = int(len(np.unique(ds_recortado["seed"])))

        edges_balanceado = meta.get("fitness", {}).get("edges")
        if edges_balanceado is not None:
            meta["distribucion_bins"]["despues"] = contar_bins_con_edges(
                ds_recortado["fitness"], edges_balanceado
            )
        
        meta["alineamiento_global"] = {
            "aplicado": True,
            "criterio_activacion": "todos_los_datasets_con_bins_cuantiles",
            "tipos_bins_detectados": list(tipos_bins),
            "n_original_balanceado": n_actual,
            "n_alineado": tam_minimo,
            "comentario": (
                "Recorte adicional del dataset mayor para igualarlo al menor. "
                "Solo se aplica porque todos los datasets comparados usan bins por cuantiles."
            ),
        }
        
        meta["muestras"]["balanceado_pre_alineacion"] = n_actual
        meta["muestras"]["retencion_pct_pre_alineacion"] = meta["muestras"].get("retencion_pct")
        meta["muestras"]["balanceado"] = tam_minimo
        meta["muestras"]["alineado"] = tam_minimo
        meta["muestras"]["retencion_pct"] = round((tam_minimo / meta["muestras"]["original"]) * 100, 4)
        meta["muestras"]["retencion_pct_alineado"] = round((tam_minimo / meta["muestras"]["original"]) * 100, 4)
        
        # Re-escribir archivos pero prefijados para no pisar el raw si lo necesitamos luego, 
        # aunque sobreescribir asegura compatibilidad. Vamos a sobreescribir.
        guardar_dataset(ds_recortado, path_npz, meta)
        print(f" -> Guardado exitosamente.")

if __name__ == "__main__":
    procesar_alineamiento()
