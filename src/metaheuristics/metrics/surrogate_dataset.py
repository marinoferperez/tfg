"""
Dataset acumulativo de evaluaciones reales de la función objetivo.

Almacena cada evaluación (eval_id, x, fitness) junto con rangos por generación
y métricas de diversidad. Es la fuente de datos para los modelos subrogados offline.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

class SurrogateDataset:
    """
    Acumula las evaluaciones reales producidas durante una ejecución de la metaheurística.

    Cada llamada a registrar_evaluacion registra un par (x, fitness). Al finalizar
    la ejecución, guardar_dataset_hdf5 almacena el dataset en HDF5.
    """

    def __init__(self, algoritmo, problema, seed, run_info=None):
        """
        algoritmo: nombre de la metaheurística.
        problema: nombre del benchmark.
        seed: semilla de la ejecución.
        run_info: dict con metadatos adicionales del experimento (funcid, dim).
        """
        self.algoritmo = str(algoritmo)
        self.problema = str(problema)
        self.seed = int(seed)
        self.run_info = dict(run_info) if isinstance(run_info, dict) else {}

        self.filas = []
        self._rangos_generacion = {}

    def registrar_evaluacion(self, eval_id, generacion=None, x=None, fitness=None):
        """
        Registra una evaluación real en el dataset.

        eval_id: identificador secuencial de la evaluación.
        generacion: generación en la que se produjo la evaluación.
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

        if generacion is not None:
            gen = int(generacion)
            rango = self._rangos_generacion.get(gen)
            payload = {
                "eval_id_start": int(eval_id),
                "eval_id_end": int(eval_id),
            }
            if rango is None:
                self._rangos_generacion[gen] = payload
            else:
                rango["eval_id_end"] = int(eval_id)

    def obtener_rangos_generacion(self):
        """
        Devuelve un dict {generacion: {eval_id_inicio, eval_id_fin}} ordenado
        por número de generación.
        """
        salida = {}
        for gen, rango in sorted(self._rangos_generacion.items(), key=lambda kv: kv[0]):
            salida[int(gen)] = {
                "eval_id_inicio": int(rango["eval_id_start"]),
                "eval_id_fin": int(rango["eval_id_end"]),
            }
        return salida

    def calcular_diversidad_rango(self, eval_id_inicio, eval_id_fin):
        """
        Calcula la diversidad euclidea media sobre las evaluaciones en [eval_id_inicio, eval_id_fin].

        eval_id_inicio: primer eval_id del rango (inclusivo).
        eval_id_fin: último eval_id del rango (inclusivo).

        Retorna un dict con div_dist_euclidea y div_dist_euclidea_normalizada, o None
        si no hay suficientes puntos.
        """
        try:
            inicio = int(eval_id_inicio)
            fin = int(eval_id_fin)
        except (TypeError, ValueError):
            return None
        if fin < inicio:
            return None

        puntos = []
        for fila in self.filas:
            eval_id = int(fila.get("eval_id", -1))
            if inicio <= eval_id <= fin and "x" in fila:
                puntos.append(np.asarray(fila["x"], dtype=float))

        if len(puntos) < 2:
            return None

        poblacion = np.asarray(puntos, dtype=float)
        if poblacion.ndim != 2 or poblacion.shape[0] < 2:
            return None

        centroide = np.mean(poblacion, axis=0)
        dists = np.linalg.norm(poblacion - centroide, axis=1)
        div = float(np.mean(dists))

        dimension = int(poblacion.shape[1])
        if dimension <= 0:
            div_norm = float("nan")
        else:
            div_norm = div / dimension

        return {
            "div_dist_euclidea": div,
            "div_dist_euclidea_normalizada": float(div_norm),
        }

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

    def _construir_dataframe(self):
        """Construye el DataFrame con eval_id, fitness y coordenadas x listo para HDF5."""
        muestras_df = pd.DataFrame(
            {
                "eval_id": [int(f.get("eval_id", 0)) for f in self.filas],
                "fitness": [float(f.get("fitness", float("nan"))) for f in self.filas],
            }
        )

        if self.filas:
            x_data = np.asarray([np.asarray(f.get("x", []), dtype=float) for f in self.filas], dtype=np.float64)
            x_df = pd.DataFrame(x_data, columns=[f"x_{i}" for i in range(x_data.shape[1])])
            return pd.concat([muestras_df, x_df], axis=1)

        return muestras_df


def guardar_dataset_hdf5(dataset, ruta_base):
    """
    Almacena el dataset en HDF5 dentro de ruta_base.

    dataset: instancia de SurrogateDataset con la ejecución completada.
    ruta_base: directorio de salida.

    Retorna un dict con la ruta al archivo HDF5 generado.
    """
    dir_salida = Path(ruta_base)
    dir_salida.mkdir(parents=True, exist_ok=True)

    base_path = dir_salida / (
        f"dataset_{dataset._normaliza_fragmento(dataset.algoritmo)}"
        f"_{dataset._normaliza_fragmento(dataset.problema)}"
        f"_{dataset._identificador_dataset()}"
    )
    dataset_hdf5_path = base_path.with_suffix(".h5")
    df = dataset._construir_dataframe()

    try:
        with pd.HDFStore(dataset_hdf5_path, mode="w", complib="zlib", complevel=6) as store:
            store.put("dataset", df, format="table")
    except ImportError as exc:
        raise ImportError(
            "No se pudo escribir el dataset en HDF5 porque pandas.HDFStore "
            "requiere la dependencia 'tables' (PyTables)."
        ) from exc

    return {"dataset_hdf5": str(dataset_hdf5_path)}
