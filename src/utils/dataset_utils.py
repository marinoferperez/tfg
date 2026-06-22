"""
Utilidades para cargar datasets HDF5 generados por las metaheurísticas.
"""

import json
from pathlib import Path

import numpy as np


def leer_seed_de_resumen(ruta_resumen):
    """
    Lee la semilla de un experimento desde su resumen.json.

    ruta_resumen: ruta al archivo resumen.json generado por RecolectorMetricasDEAP.
    """
    ruta_resumen = Path(ruta_resumen)
    
    if not ruta_resumen.is_file():
        raise FileNotFoundError(f"No se encontró resumen.json: {ruta_resumen}")
    with ruta_resumen.open("r", encoding="utf-8") as f:
        data = json.load(f)
        
    return int(data["metadata"]["seed"])


def inferir_seed(ruta):
    """
    Infiere la semilla de una run buscando resumen.json en el mismo directorio.

    ruta: ruta al dataset .h5 de la run.
    """
    return leer_seed_de_resumen(Path(ruta).parent / "resumen.json")


def cargar_dataset(ruta):
    """
    Carga un dataset HDF5 e inyecta la columna seed si falta.

    ruta: ruta al archivo .h5 o .hdf5 del dataset.

    Retorna un dict.
    """
    ruta = Path(ruta)
    
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el dataset: {ruta}")
    
    dataset = _cargar_hdf5(ruta)
    
    if "seed" not in dataset:
        seed = inferir_seed(ruta)
        dataset["seed"] = np.full(len(dataset["fitness"]), seed, dtype=np.int32)
        
    return dataset


def _cargar_hdf5(ruta):
    """
    Carga un dataset desde un fichero HDF5 generado por SurrogateDataset.

    ruta: ruta al archivo .h5 o .hdf5.

    Retorna un dict con columnas individuales y x como array 2D (columnas x_*).
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("Para cargar datasets HDF5 necesitas pandas instalado.") from exc

    with pd.HDFStore(ruta, mode="r") as store:
        df = store["dataset"]

    dataset = {col: df[col].to_numpy() for col in df.columns}
    x_cols = sorted(
        [col for col in df.columns if col.startswith("x_")],
        key=lambda c: int(c.split("_")[1]),
    )
    
    if x_cols:
        dataset["x"] = df[x_cols].to_numpy(dtype=float)
        
    return dataset
