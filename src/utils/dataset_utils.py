"""
Utilidades para cargar, filtrar y guardar datasets generados por las metaheurísticas.

Los datasets se almacenan como archivos .npz (NumPy comprimido) o HDF5. Cada dataset
contiene al menos las columnas eval_id, fitness, x y seed. Las funciones de este módulo
se usan en el pipeline de evaluación offline de modelos subrogados.
"""

import json
from pathlib import Path

import numpy as np

from src.utils.file_io import escribir_json


# ---------------------------------------------------------------------------
# Inferencia de seed
# ---------------------------------------------------------------------------

def leer_seed_de_resumen(path_resumen_json):
    """
    Lee la semilla de un experimento desde su resumen.json.

    path_resumen_json: ruta al archivo resumen.json generado por RecolectorMetricasDEAP.
    """
    path_resumen_json = Path(path_resumen_json)

    if not path_resumen_json.is_file():
        raise FileNotFoundError(f"No se encontró resumen.json: {path_resumen_json}")

    with path_resumen_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return int(data["metadata"]["seed"])

def inferir_seed(path_npz):
    """
    Infiere la semilla de una run buscando resumen.json en el mismo directorio.

    path_npz: ruta al dataset .npz o .h5 de la run.
    """
    path_resumen = Path(path_npz).parent / "resumen.json"
    return leer_seed_de_resumen(path_resumen)


# ---------------------------------------------------------------------------
# Carga de datasets
# ---------------------------------------------------------------------------

def cargar_dataset(path_npz):
    """
    Carga un dataset desde .npz o HDF5 e inyecta la columna seed si falta.

    path_npz: ruta al archivo dataset. Si termina en .h5 o .hdf5, se usa HDF5.

    Retorna un dict {columna: array_numpy}.
    """
    path_npz = Path(path_npz)
    if not path_npz.is_file():
        raise FileNotFoundError(f"No se encontró el dataset: {path_npz}")

    if path_npz.suffix in {".h5", ".hdf5"}:
        dataset = cargar_dataset_desde_hdf5(path_npz)
    else:
        data = np.load(path_npz, allow_pickle=True)
        dataset = {clave: np.asarray(data[clave]) for clave in data.files}

    n = len(dataset["fitness"])

    if "seed" not in dataset:
        seed = inferir_seed(path_npz)
        dataset["seed"] = np.full(n, seed, dtype=np.int32)

    return dataset


def cargar_dataset_desde_hdf5(path_hdf5):
    """
    Carga un dataset desde un fichero HDF5 generado por SurrogateDataset.

    path_hdf5: ruta al archivo .h5 o .hdf5.

    Retorna un dict con columnas individuales y x como array 2D si hay columnas x_*.
    """
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


