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
    def __init__(self):
        # con statitics se recogen estadisticas
        self.stats = tools.Statistics(lambda x: x)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)
        self.stats.register("promedio", np.mean)
        self.stats.register("desv_std", np.std)
        self.stats.register("mediana", np.median)

        # logbook actua como una lista de diccionarios
        self.logbook = tools.Logbook()
        self.logbook.header = [
            "evaluaciones", "generacion", "tam_poblacion", "tam_poblacion_sin_inf", "num_inf", "min (mejor_fitness)", "promedio", "desv_std",
            "mediana", "max (peor_fitness)", "mejor_hasta_ahora", "tiempo_s",
        ]
        self._mejor_hasta_ahora = None

    # registrar añade una entrada de metricas al logbook
    # tomando el vector de fitness de la poblacion actual
    # --------------------------------------------------
    # * fitness = array de fitness de cada individuo de la pob actual
    # * evaluaciones = cuantas veces se ha evaluado la func objetivo hasta ese momento (evals)
    # * generacion
    # * tiempo_s = tiempo transcurrido hasta este momento

    def registrar(self, generacion, fitness, evaluaciones, tiempo_s, tam_poblacion = None, mejor_hasta_ahora = None):
        fitness = np.asarray(fitness, dtype=float).reshape(-1)

        if tam_poblacion is None:
            tam_poblacion = int(fitness.size)

        # limpieza de posibles infs para DE 
        num_inf = int(np.isinf(fitness).sum())
        fitness_sin_inf = fitness[np.isfinite(fitness)]
        tam_poblacion_sin_inf = int(fitness_sin_inf.size)

        # si el tamaño es nulo -> no hay fitness finitos 
        if fitness_sin_inf.size == 0: fitness_sin_inf = np.array([float("inf")])

        registro = self.stats.compile(fitness_sin_inf)
        mejor_actual = float(np.min(fitness_sin_inf)) # minimizacion

        if self._mejor_hasta_ahora is None:
            self._mejor_hasta_ahora = mejor_actual
        else:
            self._mejor_hasta_ahora = min(self._mejor_hasta_ahora, mejor_actual)

        if mejor_hasta_ahora is None:
            mejor_hasta_ahora = self._mejor_hasta_ahora

        # con .record DEAP añade una nueva entrada/diccionario a la lista con los distintos elementos indicados
        self.logbook.record(
            evaluaciones = int(evaluaciones),
            generacion = int(generacion),
            tam_poblacion = int(tam_poblacion),
            tam_poblacion_sin_inf = int(fitness_sin_inf.size), # ind con fit infinito que no han contribuido en el calc de métricas
            num_inf = int(num_inf),                           # num indiv con fitness inf
            min = float(registro["min"]),                      # mejor fitness en la pob actual
            promedio = float(registro["promedio"]),            # media del fitness de la pob -> mide calidad global
            desv_std = float(registro["desv_std"]),            # dispersion del fitness -> std baja + min no mejora -> estancamiento
            mediana = float(registro["mediana"]),              # valor tipico de la pob -> robusto a outliers   
            max = float(registro["max"]),                      # peor fitness 
            mejor_hasta_ahora = float(mejor_hasta_ahora),      # mejor fit encontrado hasta ese momento -> en la ultima it = min
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
        # minimos = [float(fila["min"]) for fila in historial]

        hubo_inf = any(int(f.get("num_inf", 0)) > 0 for f in historial)
        total_inf = sum(int(f.get("num_inf", 0)) for f in historial)

        return {
            "generaciones_registradas": int(len(historial)),
            "evaluaciones_totales": int(ultimo["evaluaciones"]),
            "min_final": float(ultimo["min"]),
            "promedio_final": float(ultimo["promedio"]),
            "desv_std_final": float(ultimo["desv_std"]),
            "mediana_final": float(ultimo["mediana"]),
            "max_final": float(ultimo["max"]),
            "mejor_hasta_ahora (fitness)": float(ultimo["mejor_hasta_ahora"]),
            "tiempo_total_s": float(ultimo["tiempo_s"]),
            "hubo_inf": bool(hubo_inf),
            "total_inf": int(total_inf)
        }

    # guardar_csv_json 
    def guardar_csv_json(self, ruta_base, metadata=None):
        base = Path(ruta_base)
        base.mkdir(parents=True, exist_ok=True)

        historial = self.obtener_logbook()
        resumen = self.obtener_resumen_final()
        payload_resumen = {
            "metadata": metadata if metadata is not None else {},
            "resumen": resumen,
        }

        ruta_logbook_csv = base / "logbook.csv"
        ruta_resumen_json = base / "resumen.json"

        with ruta_logbook_csv.open("w", encoding="utf-8", newline="") as f_csv:
            campos = list(self.logbook.header)

            writer = csv.DictWriter(f_csv, fieldnames=campos)
            writer.writeheader()
            for fila in historial:
                writer.writerow(fila)

        with ruta_resumen_json.open("w", encoding="utf-8") as f_resumen:
            json.dump(payload_resumen, f_resumen, ensure_ascii=False, indent=2)

        return {
            "logbook_csv": str(ruta_logbook_csv),
            "resumen_json": str(ruta_resumen_json),
        }
