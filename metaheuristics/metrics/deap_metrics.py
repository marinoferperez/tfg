# utilidades de metrica basadas en DEAP (Statistics + Logbook).

import csv
import json
from pathlib import Path
import numpy as np
from deap import tools

class RecolectorMetricasDEAP:
    # constructor del recolector de metricas
    # -------------------------------------
    # se registra metricas por generacion
    def __init__(
        self,
        seed = 42,
        k_pares_hamming = 200,
    ):
        # con statitics se recogen estadisticas
        self.stats = tools.Statistics(lambda x: x)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)
        self.stats.register("promedio", np.mean)
        self.stats.register("desv_std", np.std)
        self.stats.register("mediana", np.median)

        # logbook actua como una lista de diccionarios
        self.logbook = tools.Logbook()
        self._cabecera_base = [
            "evaluaciones",
            "generacion",
            "tam_poblacion",
            "min/mejor_hasta_ahora",
            "promedio",
            "desv_std",
            "mediana",
            "max",
            "tiempo_s",
        ]
        self.logbook.header = list(self._cabecera_base)
        self._diversidad_por_generacion = {}
        # k fijo para comparabilidad entre algoritmos/runs en QAP.
        # Se ignora cualquier valor distinto recibido por parámetro.
        self._k_pares_hamming = int(k_pares_hamming)
        self._rng = np.random.default_rng(seed) 
        

    # ---------- diversidad (continuo) ----------

    # _diversidad_dist_euclidea mide como de lejos estan los individuos del centro de la poblacion (centroide)
    # si todos estan muy cerca del centroide -> baja diversidad en la pob

    def _diversidad_dist_euclidea(self, poblacion):
        centroide = np.mean(poblacion, axis=0)
        dists = np.linalg.norm(poblacion - centroide, axis=1)
        return float(np.mean(dists))

    # ---------- diversidad (permutaciones) ----------

    # _diversidad_media_hamming (normalizada) mide cuantas posiciones difieren entre dos sol (permutaciones)
    # si las perm son muy parecidas -> hamming baja -> baja diversidad

    def _hamming(self, a, b):
        a_arr = np.asarray(a)
        b_arr = np.asarray(b)
        if np.issubdtype(a_arr.dtype, np.floating) or np.issubdtype(b_arr.dtype, np.floating):
            distintos = ~np.isclose(a_arr, b_arr, rtol=0.0, atol=1e-12, equal_nan=True)
            return int(np.sum(distintos))
        return int(np.sum(a_arr != b_arr))

    def _diversidad_media_hamming(self, permuts, k_pares):
        # permuts: (N, n) int
        if permuts.ndim != 2 or permuts.shape[0] < 2:
            return float("nan")

        N, n = permuts.shape
        if n <= 0:
            return float("nan")
        # número de pares posibles
        total_pairs = N * (N - 1) // 2
        if total_pairs <= 0:
            return float("nan")

        k = min(int(k_pares), int(total_pairs))
        if k <= 0:
            return float("nan")

        # si se piden todos los pares, devolvemos media exacta
        if k == total_pairs:
            hs = []
            for i in range(N - 1):
                for j in range(i + 1, N):
                    hs.append(self._hamming(permuts[i], permuts[j]))
            return (float(np.mean(hs)) / float(n)) if hs else float("nan")

        # si se piden menos pares, muestreamos pares unicos (i < j)
        pares = set()
        while len(pares) < k:
            i, j = self._rng.choice(N, size=2, replace=False)
            if i > j:
                i, j = j, i
            pares.add((int(i), int(j)))

        hs = [self._hamming(permuts[i], permuts[j]) for i, j in pares]
        return (float(np.mean(hs)) / float(n)) if hs else float("nan")

    # registrar añade una entrada de metricas al logbook
    # tomando el vector de fitness de la poblacion actual
    # --------------------------------------------------
    # * fitness = array de fitness de cada individuo de la pob actual
    # * evaluaciones = cuantas veces se ha evaluado la func objetivo hasta ese momento (evals)
    # * generacion
    # * tiempo_s = tiempo transcurrido hasta este momento

    def registrar(
        self,
        generacion,
        fitness = None,
        evaluaciones = None,
        tiempo_s = None,
        tam_poblacion = None,
        mejor_hasta_ahora = None,
        fitness_vector = None,
        poblacion = None,
        permutaciones = None,
        vectores_hamming = None,
    ):
        # compatibilidad: algunas llamadas antiguas usan fitness_vector
        if fitness is None:
            fitness = fitness_vector
        if fitness is None:
            raise ValueError("Debes pasar 'fitness' o 'fitness_vector' a registrar().")
        if evaluaciones is None:
            raise ValueError("Debes pasar 'evaluaciones' a registrar().")
        if tiempo_s is None:
            raise ValueError("Debes pasar 'tiempo_s' a registrar().")

        fitness = np.asarray(fitness, dtype=float).reshape(-1)

        if tam_poblacion is None:
            tam_poblacion = int(fitness.size)

        # limpieza de posibles infs para DE 
        fitness_sin_inf = fitness[np.isfinite(fitness)]

        # si el tamaño es nulo -> no hay fitness finitos 
        if fitness_sin_inf.size == 0: fitness_sin_inf = np.array([float("inf")])

        registro = self.stats.compile(fitness_sin_inf)

        div_dist_euclidea = float('nan')
        div_media_hamming = float('nan')
        registro_div = {}

        # La métrica de diversidad se infiere por el tipo de representación
        # que recibe el recolector en cada algoritmo:
        # continuo -> distancia euclídea, combinatorio -> Hamming media.
        base_hamming = None
        if vectores_hamming is not None:
            base_hamming = np.asarray(vectores_hamming)
        elif permutaciones is not None:
            base_hamming = np.asarray(permutaciones)

        if base_hamming is not None and base_hamming.ndim == 2 and base_hamming.shape[0] >= 2:
            div_media_hamming = self._diversidad_media_hamming(base_hamming, k_pares=self._k_pares_hamming)
            registro_div["div_media_hamming"] = float(div_media_hamming)
        elif poblacion is not None:
            pop = np.asarray(poblacion)
            if pop.ndim == 2 and pop.shape[0] >= 2:
                pop_f = pop.astype(float, copy=False)
                mask = np.all(np.isfinite(pop_f), axis=1)
                pop_f = pop_f[mask]
                if pop_f.shape[0] >= 2:
                    div_dist_euclidea = self._diversidad_dist_euclidea(pop_f)
                    registro_div["div_dist_euclidea"] = float(div_dist_euclidea)

        if registro_div:
            self._diversidad_por_generacion[int(generacion)] = dict(registro_div)

        self.logbook.record(
            evaluaciones = int(evaluaciones),
            generacion = int(generacion),
            tam_poblacion = int(tam_poblacion),
            **{"min/mejor_hasta_ahora": float(registro["min"])}, # minimizacion
            promedio = float(registro["promedio"]),            # media del fitness de la pob -> mide calidad global
            desv_std = float(registro["desv_std"]),            # dispersion del fitness -> std baja + min no mejora -> estancamiento
            mediana = float(registro["mediana"]),              # valor tipico de la pob -> robusto a outliers   
            max = float(registro["max"]),                      # peor fitness 
            tiempo_s = float(tiempo_s),
        )

    # _serializa_valor convierte tipos numpy a tipos nativos de Python
    # def _serializa_valor(self, valor):
    #     if isinstance(valor, np.generic):
    #         return valor.item()
    #     if isinstance(valor, np.ndarray):
    #         return valor.tolist()
    #     return valor

    # obtener_logbook devuelve el historial en una lista de diccionarios
    # apta para guardar en json o csv
    def obtener_logbook(self):
        return [dict(e) for e in self.logbook]
        # historial = []
        # for entrada in self.logbook:
        #     fila = {}
        #     for clave, valor in dict(entrada).items():
        #         fila[clave] = self._serializa_valor(valor)
        #     historial.append(fila)
        # return historial

    # alias retrocompatible
    def obtener_logbook_serializable(self):
        return self.obtener_logbook()

    def obtener_diversidad_por_generacion(self):
        return dict(sorted(self._diversidad_por_generacion.items(), key=lambda kv: kv[0]))

    # obtener_resumen_final devuelve un resumen de la ultima generaion
    # y algunos agregados utiles para analizar la ejecucion
    def obtener_resumen_final(self):
        historial = self.obtener_logbook()
        if len(historial) == 0:
            return {
                "generaciones_registradas": 0,
                "evaluaciones_totales": 0,
            }

        ultimo = historial[-1] # ultima generacion de pob
        resumen = {
            "generaciones_registradas": int(len(historial)),
            "evaluaciones_totales": int(ultimo["evaluaciones"]),
            "min_final": float(ultimo["min/mejor_hasta_ahora"]),
            "promedio_final": float(ultimo["promedio"]),
            "desv_std_final": float(ultimo["desv_std"]),
            "mediana_final": float(ultimo["mediana"]),
            "max_final": float(ultimo["max"]),
            "mejor_hasta_ahora": float(ultimo["min/mejor_hasta_ahora"]),
            "tiempo_total_s": float(ultimo["tiempo_s"]),
        }
        return resumen

    # guardar_csv_json 
    def guardar_csv_json(self, ruta_base, metadata=None):
        base = Path(ruta_base)
        base.mkdir(parents=True, exist_ok=True)

        historial = self.obtener_logbook()
        resumen = self.obtener_resumen_final()
        metadata_limpia = {}
        if metadata is not None:
            metadata_limpia = {k: v for k, v in metadata.items() if v is not None}
        payload_resumen = {
            "metadata": metadata_limpia,
            "resumen": resumen,
        }

        ruta_logbook_csv = base / "logbook.csv"
        ruta_resumen_json = base / "resumen.json"

        with ruta_logbook_csv.open("w", encoding="utf-8", newline="") as f_csv:
            campos = list(self.logbook.header)
            writer = csv.DictWriter(f_csv, fieldnames=campos, lineterminator="\n")
            writer.writeheader()
            writer.writerows(historial)

        with ruta_resumen_json.open("w", encoding="utf-8") as f_resumen:
            json.dump(payload_resumen, f_resumen, ensure_ascii=False, indent=2)

        return {
            "logbook_csv": str(ruta_logbook_csv),
            "resumen_json": str(ruta_resumen_json),
        }
