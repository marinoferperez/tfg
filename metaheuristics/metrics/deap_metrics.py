# utilidades de metrica basadas en DEAP (Statistics + Logbook).

from pathlib import Path
import numpy as np
from deap import tools

from preprocesado_de_datos.utils.path_utils import escribir_csv_dicts, escribir_json

class RecolectorMetricasDEAP:
    # constructor del recolector de metricas
    # -------------------------------------
    # se registra metricas por generacion
    def __init__(
        self,
        seed = 42,
        filtrar_evals_no_crecientes = False,
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
        self._rng = np.random.default_rng(seed)
        self._rangos_generacion = {}
        self._filtrar_evals_no_crecientes = bool(filtrar_evals_no_crecientes)
        self._ultima_eval_registrada = None
        

    # ---------- diversidad (continuo) ----------

    # _diversidad_dist_euclidea mide como de lejos estan los individuos del centro de la poblacion (centroide)
    # si todos estan muy cerca del centroide -> baja diversidad en la pob

    def _diversidad_dist_euclidea(self, poblacion):
        centroide = np.mean(poblacion, axis=0)
        dists = np.linalg.norm(poblacion - centroide, axis=1)
        return float(np.mean(dists))

    def _diversidad_dist_euclidea_normalizada(self, diversidad, dimension, rango_inf=-100.0, rango_sup=100.0):
        dimension = int(dimension)
        if dimension <= 0:
            return float("nan")
        return float(diversidad) / dimension

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
        sobrescribir_ultima = False,
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

        if sobrescribir_ultima and len(self.logbook) > 0:
            self.logbook.pop(-1)
            if len(self.logbook) > 0:
                self._ultima_eval_registrada = int(self.logbook[-1]["evaluaciones"])
            else:
                self._ultima_eval_registrada = None

        if self._filtrar_evals_no_crecientes and self._ultima_eval_registrada is not None:
            if int(evaluaciones) <= int(self._ultima_eval_registrada):
                return

        registro = self.stats.compile(fitness_sin_inf)

        registro_div = {}

        if poblacion is not None:
            pop = np.asarray(poblacion)
            if pop.ndim == 2 and pop.shape[0] >= 2:
                pop_f = pop.astype(float, copy=False)
                mask = np.all(np.isfinite(pop_f), axis=1)
                pop_f = pop_f[mask]
                if pop_f.shape[0] >= 2:
                    div = float(self._diversidad_dist_euclidea(pop_f))
                    registro_div["div_dist_euclidea"] = div
                    registro_div["div_dist_euclidea_normalizada"] = float(
                        self._diversidad_dist_euclidea_normalizada(div, pop_f.shape[1])
                    )

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
        self._ultima_eval_registrada = int(evaluaciones)

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

    def anotar_rangos_generacion(self, rangos_generacion):
        self._rangos_generacion = dict(sorted(dict(rangos_generacion).items(), key=lambda kv: kv[0]))

    def anotar_diversidad_generacion(self, generacion, payload):
        try:
            gen = int(generacion)
        except (TypeError, ValueError):
            return
        if not isinstance(payload, dict):
            return

        limpio = {}
        if "div_dist_euclidea" in payload and payload["div_dist_euclidea"] is not None:
            limpio["div_dist_euclidea"] = float(payload["div_dist_euclidea"])
        if "div_dist_euclidea_normalizada" in payload and payload["div_dist_euclidea_normalizada"] is not None:
            limpio["div_dist_euclidea_normalizada"] = float(payload["div_dist_euclidea_normalizada"])
        if limpio:
            self._diversidad_por_generacion[gen] = limpio

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

        identificador = None
        if metadata_limpia:
            algoritmo = str(metadata_limpia.get("algoritmo", "mh")).strip().lower()
            problema = str(metadata_limpia.get("problema", "problema")).strip().lower()
            if problema == "cec2017":
                funcid = metadata_limpia.get("funcid")
                dim = metadata_limpia.get("dim")
                if funcid is not None and dim is not None:
                    identificador = f"{algoritmo}_{problema}_f{int(funcid)}_d{int(dim)}"
        if identificador is None:
            identificador = "mh_resultados"

        ruta_logbook_csv = base / f"resultados_{identificador}.csv"
        ruta_resumen_json = base / "resumen.json"

        historial_enriquecido = []
        for entrada in historial:
            fila = dict(entrada)
            gen = int(fila["generacion"])
            rango = self._rangos_generacion.get(gen, {})
            div = self._diversidad_por_generacion.get(gen, {})
            fila["eval_id_inicio"] = rango.get("eval_id_inicio")
            fila["eval_id_fin"] = rango.get("eval_id_fin")
            fila["div_dist_euclidea"] = div.get("div_dist_euclidea")
            fila["div_dist_euclidea_normalizada"] = div.get("div_dist_euclidea_normalizada")
            historial_enriquecido.append(fila)

        fieldnames = list(self.logbook.header) + [
            "eval_id_inicio",
            "eval_id_fin",
            "div_dist_euclidea",
            "div_dist_euclidea_normalizada",
        ]
        escribir_csv_dicts(ruta_logbook_csv, historial_enriquecido, fieldnames=fieldnames)
        escribir_json(ruta_resumen_json, payload_resumen)

        return {
            "resultados_csv": str(ruta_logbook_csv),
            "resumen_json": str(ruta_resumen_json),
        }
