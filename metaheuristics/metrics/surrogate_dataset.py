import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


class SurrogateDataset:
    # guarda todas las evals reales (x --> fitness)
    # util para entrenar surrogates (online/offline)

    def __init__(self, algoritmo, problema, seed, run_info = None):
        self.algoritmo = str(algoritmo)
        self.problema = str(problema)
        algoritmo_norm = self.algoritmo.strip().lower()
        problema_norm = self.problema.strip().lower()
        self.seed = int(seed)
        self.run_info = dict(run_info) if isinstance(run_info, dict) else {}

        self._guardar_x = True

        self.filas = []
        self._rangos_generacion = {}
        self._diversidad_generacion = []

    def individuo_to_dataset(self, eval_id, generacion=None, x=None, fitness=None, perm = None):
        fila = {
            "eval_id": int(eval_id),
            "fitness": float(fitness),
        }
        if self._guardar_x and x is not None:
            fila["x"] = np.asarray(x, dtype=float).tolist()

        self.filas.append(fila)

        if generacion is not None:
            gen = int(generacion)
            rango = self._rangos_generacion.get(gen)
            idx = len(self.filas) - 1
            payload = {
                "row_start": idx,
                "row_end": idx,
                "eval_id_start": int(eval_id),
                "eval_id_end": int(eval_id),
            }
            if rango is None:
                self._rangos_generacion[gen] = payload
            else:
                rango["row_end"] = idx
                rango["eval_id_end"] = int(eval_id)

    def anotar_diversidad_por_generacion(self, diversidad_por_generacion):
        self._diversidad_generacion = []
        if not diversidad_por_generacion:
            return

        diversidad_limpia = {}
        for gen, payload in dict(diversidad_por_generacion).items():
            try:
                gen_int = int(gen)
            except (TypeError, ValueError):
                continue
            if not isinstance(payload, dict):
                continue

            limpio = {}
            if "div_dist_euclidea" in payload:
                limpio["div_dist_euclidea"] = float(payload["div_dist_euclidea"])
            if "div_dist_euclidea_normalizada" in payload:
                limpio["div_dist_euclidea_normalizada"] = float(payload["div_dist_euclidea_normalizada"])
            if limpio:
                diversidad_limpia[gen_int] = limpio

        if not diversidad_limpia:
            return

        for gen, extras in sorted(diversidad_limpia.items(), key=lambda kv: kv[0]):
            rango = self._rangos_generacion.get(gen)
            if rango is None:
                continue
            fila = {
                "generacion": int(gen),
                "eval_id_inicio": int(rango["eval_id_start"]),
                "eval_id_fin": int(rango["eval_id_end"]),
            }
            fila.update(extras)
            self._diversidad_generacion.append(fila)

    def obtener_rangos_generacion(self):
        salida = {}
        for gen, rango in sorted(self._rangos_generacion.items(), key=lambda kv: kv[0]):
            salida[int(gen)] = {
                "eval_id_inicio": int(rango["eval_id_start"]),
                "eval_id_fin": int(rango["eval_id_end"]),
            }
        return salida

    def calcular_diversidad_rango(self, eval_id_inicio, eval_id_fin, rango_inf=-100.0, rango_sup=100.0):
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
        txt = str(valor).strip().lower()
        txt = re.sub(r"[^a-z0-9_-]+", "_", txt)
        txt = re.sub(r"_+", "_", txt).strip("_")
        return txt or "na"

    def _identificador_dataset(self):
        if self.problema == "cec2017":
            funcid = self.run_info.get("funcid")
            dim = self.run_info.get("dim")
            if funcid is not None and dim is not None:
                return f"f{int(funcid)}_d{int(dim)}"
            if funcid is not None:
                return f"f{int(funcid)}"
            return "cec"
        return self._normaliza_fragmento(self.run_info.get("id", "run"))

    def _construir_componentes(self):
        muestras_df = pd.DataFrame(
            {
                "eval_id": [int(f.get("eval_id", 0)) for f in self.filas],
                "fitness": [float(f.get("fitness", float("nan"))) for f in self.filas],
            }
        )

        x_df = None
        if self._guardar_x:
            if self.filas:
                x_data = np.asarray([np.asarray(f.get("x", []), dtype=float) for f in self.filas], dtype=np.float64)
                x_df = pd.DataFrame(x_data, columns=[f"x_{i}" for i in range(x_data.shape[1])])
            else:
                x_df = pd.DataFrame()

        diversidad_df = pd.DataFrame(self._diversidad_generacion)
        metadata_df = pd.DataFrame(
            [
                {
                    "algoritmo": self.algoritmo,
                    "problema": self.problema,
                    "seed": int(self.seed),
                    "run_info_json": json.dumps(self.run_info, ensure_ascii=True, sort_keys=True),
                }
            ]
        )

        return muestras_df, x_df, diversidad_df, metadata_df

    def guardar_csv_json(self, ruta_base):
        dir_salida = Path(ruta_base)
        dir_salida.mkdir(parents=True, exist_ok=True)

        base_path = dir_salida / f"dataset_{self._normaliza_fragmento(self.algoritmo)}_{self._normaliza_fragmento(self.problema)}_{self._identificador_dataset()}"
        dataset_hdf5_path = base_path.with_suffix(".h5")
        muestras_df, x_df, diversidad_df, metadata_df = self._construir_componentes()
        dataset_hdf_df = muestras_df.copy()
        if x_df is not None and not x_df.empty:
            dataset_hdf_df = pd.concat([dataset_hdf_df, x_df], axis=1)

        try:
            with pd.HDFStore(dataset_hdf5_path, mode="w", complib="zlib", complevel=6) as store:
                store.put("dataset", dataset_hdf_df, format="table")
        except ImportError as exc:
            raise ImportError(
                "No se pudo escribir el dataset en HDF5 porque pandas.HDFStore "
                "requiere la dependencia 'tables' (PyTables)."
            ) from exc

        return {
            "dataset_hdf5": str(dataset_hdf5_path),
            "dataset_csv": None,
            "dataset_csv_gz": None,
            "dataset_json": None,
        }
