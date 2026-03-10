import re
from pathlib import Path
import numpy as np

class SurrogateDataset:
    # guarda todas las evals reales (x --> fitness)
    # util para entrenar surrogates (online/offline)

    def __init__(self, algoritmo, problema, seed, run_info = None):
        self.algoritmo = str(algoritmo)
        self.problema = str(problema)
        self.seed = int(seed)
        self.run_info = dict(run_info) if isinstance(run_info, dict) else {}

        self.filas = []
    
    def individuo_to_dataset(self, eval_id, generacion, x, fitness, perm = None):
        fila = {
            "eval_id": int(eval_id),
            "generacion": int(generacion),
            "fitness": float(fitness),
            "x": np.asarray(x).tolist(),
        }
        if perm is not None:
            fila["perm"] = np.asarray(perm).astype(int).tolist()

        self.filas.append(fila)

    def anotar_diversidad_por_generacion(self, diversidad_por_generacion):
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
            if "div_media_hamming" in payload:
                limpio["div_media_hamming"] = float(payload["div_media_hamming"])
            if limpio:
                diversidad_limpia[gen_int] = limpio

        if not diversidad_limpia:
            return

        for fila in self.filas:
            gen = int(fila.get("generacion", -1))
            extras = diversidad_limpia.get(gen)
            if not extras:
                continue
            fila.update(extras)

    def _normaliza_fragmento(self, valor):
        txt = str(valor).strip().lower()
        txt = re.sub(r"[^a-z0-9_-]+", "_", txt)
        txt = re.sub(r"_+", "_", txt).strip("_")
        return txt or "na"

    def _identificador_dataset(self):
        if self.problema == "qap":
            instancia = self.run_info.get("instancia", "custom")
            return self._normaliza_fragmento(instancia)
        if self.problema == "cec2017":
            funcid = self.run_info.get("funcid")
            dim = self.run_info.get("dim")
            if funcid is not None and dim is not None:
                return f"f{int(funcid)}_d{int(dim)}"
            if funcid is not None:
                return f"f{int(funcid)}"
            return "cec"
        return self._normaliza_fragmento(self.run_info.get("id", "run"))

    # mantener nombre por compatibilidad: ahora genera NPZ comprimido.
    def guardar_csv_json(self, ruta_base):
        dir_salida = Path(ruta_base)
        dir_salida.mkdir(parents=True, exist_ok=True)

        dataset_npz_path = dir_salida / f"dataset_{self._normaliza_fragmento(self.algoritmo)}_{self._normaliza_fragmento(self.problema)}_{self._identificador_dataset()}.npz"

        payload = {
            "eval_id": np.asarray([int(f.get("eval_id", 0)) for f in self.filas], dtype=np.int64),
            "generacion": np.asarray([int(f.get("generacion", 0)) for f in self.filas], dtype=np.int64),
            "fitness": np.asarray([float(f.get("fitness", float("nan"))) for f in self.filas], dtype=np.float64),
        }

        if self.filas:
            payload["x"] = np.asarray([np.asarray(f.get("x", []), dtype=float) for f in self.filas], dtype=np.float64)
        else:
            payload["x"] = np.empty((0, 0), dtype=np.float64)

        incluir_perm = any("perm" in fila for fila in self.filas)
        incluir_div_euclidea = any("div_dist_euclidea" in fila for fila in self.filas)
        incluir_div_hamming = any("div_media_hamming" in fila for fila in self.filas)

        if incluir_perm:
            payload["perm"] = np.asarray([np.asarray(f.get("perm", []), dtype=int) for f in self.filas], dtype=np.int64)
        if incluir_div_euclidea:
            payload["div_dist_euclidea"] = np.asarray(
                [float(f.get("div_dist_euclidea", float("nan"))) for f in self.filas],
                dtype=np.float64,
            )
        if incluir_div_hamming:
            payload["div_media_hamming"] = np.asarray(
                [float(f.get("div_media_hamming", float("nan"))) for f in self.filas],
                dtype=np.float64,
            )

        np.savez_compressed(dataset_npz_path, **payload)
        return {
            "dataset_npz": str(dataset_npz_path),
            "dataset_csv": None,
            "dataset_csv_gz": None,
            "dataset_json": None,
        }
