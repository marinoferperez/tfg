"""
Dataset acumulativo de evaluaciones reales de la función objetivo.

Almacena cada evaluación (eval_id, x, fitness). Es la fuente de datos para los modelos subrogados offline.
"""

import re

import numpy as np
import pandas as pd

class SurrogateDataset:
    """
    Acumula las evaluaciones reales producidas durante una ejecución de la metaheurística.

    Cada llamada a registrar_evaluacion registra un par (x, fitness). Al finalizar
    la ejecución, guardar_dataset_hdf5 almacena el dataset en HDF5.
    """

    def __init__(self, algoritmo, problema, run_info=None):
        """
        algoritmo: nombre de la metaheurística.
        problema: nombre del benchmark.
        run_info: dict con metadatos adicionales del experimento (funcid, dim).
        """
        self.algoritmo = str(algoritmo)
        self.problema = str(problema)
        self.run_info = dict(run_info) if isinstance(run_info, dict) else {}

        self.filas = []

    def registrar_evaluacion(self, eval_id, x=None, fitness=None):
        """
        Registra una evaluación real en el dataset.

        eval_id: identificador secuencial de la evaluación.
        x: vector de decisión evaluado.
        fitness: valor de la función objetivo para x.
        """
        fila = {
            "eval_id": int(eval_id),
            "fitness": float(fitness),
        }

        if x is not None:
            fila["x"] = np.asarray(x, dtype=float).tolist()

        self.filas.append(fila)

    def _normaliza_fragmento(self, valor):
        """
        Convierte un valor a una cadena segura para usarla en nombres de fichero.

        valor: cualquier valor convertible a str (algoritmo, problema, run_info, …).
        """
        txt = str(valor).strip().lower()
        txt = re.sub(r"[^a-z0-9_-]+", "_", txt)
        txt = re.sub(r"_+", "_", txt).strip("_")
        return txt or "na"

    def _identificador_dataset(self):
        """Genera un identificador corto para usar en el nombre del fichero de salida."""
        if self.problema == "cec2017":
            funcid = self.run_info.get("funcid")
            dim = self.run_info.get("dim")
            
            if funcid is not None and dim is not None:
                return f"f{int(funcid)}_d{int(dim)}"
            
            if funcid is not None:
                return f"f{int(funcid)}"
        
            return "cec"
        return self._normaliza_fragmento(self.run_info.get("id", "run"))

    def construir_dataframe(self):
        """Construye el DataFrame con eval_id, fitness y coordenadas x listo para HDF5."""
        muestras_df = pd.DataFrame({"eval_id": [int(f.get("eval_id", 0)) for f in self.filas], "fitness": [float(f.get("fitness", float("nan"))) for f in self.filas]})

        if self.filas:
            x_data = np.asarray([np.asarray(f.get("x", []), dtype=float) for f in self.filas], dtype=np.float64)
            x_df = pd.DataFrame(x_data, columns=[f"x_{i}" for i in range(x_data.shape[1])])
            return pd.concat([muestras_df, x_df], axis=1)

        return muestras_df
