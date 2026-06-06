"""
utils.py – Utilidades para el preprocesado de datasets generados por las
metaheurísticas.

"""

import json
from pathlib import Path

import numpy as np

from preprocesado_de_datos.utils.path_utils import escribir_json


# ---------------------------------------------------------------------------
# 1. inferir la seed
# ---------------------------------------------------------------------------

# leer_seed_de_resumen lee la seed desde el archivo resumen.json correspondiente
# json = map, por lo que accedemos por medio de la metadata a sus parametros como seed

def leer_seed_de_resumen(path_resumen_json):
    path_resumen_json = Path(path_resumen_json)

    if not path_resumen_json.is_file():
        raise FileNotFoundError(f"No se encontró resumen.json: {path_resumen_json}")

    with path_resumen_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return int(data["metadata"]["seed"])

# inferir_seed infiere la seed de una run a partir del "resumen.json" que se encuentra en la misma carpeta

def inferir_seed(path_npz):
    path_resumen = Path(path_npz).parent / "resumen.json"
    return leer_seed_de_resumen(path_resumen)


# ---------------------------------------------------------------------------
# 2. Cargar un dataset.npz
# ---------------------------------------------------------------------------

# cargar_dataset carga un .npz y devuelve un diccionario de arrays numpy.
# .npz no contiene seed pero se añade como array con la misma long que el resto de las columnas tras ser inferida.

def cargar_dataset(path_npz):
    path_npz = Path(path_npz)
    if not path_npz.is_file():
        raise FileNotFoundError(f"No se encontró el dataset: {path_npz}")

    if path_npz.suffix in {".h5", ".hdf5"}:
        dataset = cargar_dataset_desde_hdf5(path_npz)
    else:
        data = np.load(path_npz, allow_pickle=True)
        dataset = {clave: np.asarray(data[clave]) for clave in data.files}

    n = len(dataset["fitness"])

    # se añade la seed en el dataset
    if "seed" not in dataset:
        seed = inferir_seed(path_npz)
        dataset["seed"] = np.full(n, seed, dtype=np.int32)

    return dataset


def cargar_dataset_desde_hdf5(path_hdf5):
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Para cargar datasets HDF5 necesitas pandas instalado."
        ) from exc

    with pd.HDFStore(path_hdf5, mode="r") as store:
        df = store["dataset"]

    dataset = {col: df[col].to_numpy() for col in df.columns}
    x_cols = sorted(
        [col for col in df.columns if col.startswith("x_")],
        key=lambda x: int(x.split("_")[1]),
    )
    if x_cols:
        dataset["x"] = df[x_cols].to_numpy(dtype=float)
    return dataset


# ---------------------------------------------------------------------------
# 3. Concatenar runs de un experimento
# ---------------------------------------------------------------------------

# concatenar_runs recibe una lista de rutas a .npz, carga cada uno con
# cargar_dataset (que ya inyecta la seed) y los concatena en un unico dict.
# todas las runs deben tener las mismas claves (mismo algoritmo/problema).

def concatenar_runs(lista_paths_npz):
    if not lista_paths_npz:
        raise ValueError("La lista de archivos .npz está vacía")

    runs = [cargar_dataset(npz) for npz in lista_paths_npz]

    # se comprueba que todas las runs tienen las mismas claves
    claves_ref = set(runs[0].keys())
    for i, run in enumerate(runs[1:], start=1):
        if set(run.keys()) != claves_ref:
            raise ValueError(
                f"El archivo {lista_paths_npz[i]} tiene claves distintas: "
                f"{sorted(set(run.keys()))} vs {sorted(claves_ref)}"
            )

    # concatenar cada clave
    dataset_concat = {}
    for clave in claves_ref:
        dataset_concat[clave] = np.concatenate([r[clave] for r in runs], axis=0)

    return dataset_concat


# ---------------------------------------------------------------------------
# 4. Filtrar dataset por indices
# ---------------------------------------------------------------------------

# filtrar_dataset selecciona un subconjunto de muestras del dataset
# segun los indices proporcionados (por ejemplo, los que devuelve balanceo_a_la_baja).

def filtrar_dataset(dataset, indices):
    return {clave: np.asarray(arr)[indices] for clave, arr in dataset.items()}


# ---------------------------------------------------------------------------
# 5. Guardar dataset y metadata
# ---------------------------------------------------------------------------

# guardar_dataset guarda un dataset como .npz comprimido y un .metadata.json
# con informacion util para comparaciones antes/despues del balanceo.
# el .json se guarda junto al .npz con el mismo nombre + .metadata.json

def guardar_dataset(dataset, path_salida_npz, metadata):
    path_salida_npz = Path(path_salida_npz)
    path_salida_npz.parent.mkdir(parents=True, exist_ok=True)

    dataset_npz = {
        clave: np.asarray(arr)
        for clave, arr in dataset.items()
        if clave != "div_dist_euclidea"
    }

    # guardar el dataset como .npz comprimido
    np.savez_compressed(path_salida_npz, **dataset_npz)

    metadata = dict(metadata)
    metadata["artefactos"] = {
        "dataset_npz": str(path_salida_npz),
    }

    # guardar la metadata como .json junto al .npz
    path_json = path_salida_npz.with_suffix(".metadata.json")
    escribir_json(path_json, metadata)


# ---------------------------------------------------------------------------
# 7. main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prueba rápida: carga un dataset.npz y muestra un resumen.",
    )
    parser.add_argument(
        "npz",
        type=str,
        help="Ruta al archivo dataset_*.npz a cargar.",
    )
    args = parser.parse_args()

    ds = cargar_dataset(args.npz)

    print(f"Archivo cargado: {args.npz}")
    print(f"Claves:          {sorted(ds.keys())}")
    for k, v in sorted(ds.items()):
        print(f"  {k:25s}  shape={str(v.shape):15s}  dtype={v.dtype}")
    print(f"\nSeed inferida:   {ds['seed'][0]}")
    print(f"Nº de muestras:  {len(ds['fitness'])}")
    print(f"Fitness min:     {ds['fitness'].min():.6f}")
    print(f"Fitness max:     {ds['fitness'].max():.6f}")
