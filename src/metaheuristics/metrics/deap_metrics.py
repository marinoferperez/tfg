"""
Recolector de métricas por generación basado en DEAP Statistics y Logbook.

Registra fitness, diversidad euclidea y tiempo por cada generación de la
metaheurística. Al finalizar construye un CSV de historial y un JSON de resumen.
"""

from pathlib import Path
import numpy as np
from deap import tools

from src.utils.file_io import escribir_csv_dicts, escribir_json

class RecolectorMetricasDEAP:
    """
    Acumula métricas por generación durante una ejecución de la metaheurística.

    Usa DEAP Statistics para calcular min/max/media/desv_std/mediana del fitness y
    registra la diversidad euclidea de la población cuando se proporciona.
    """

    def __init__(self, filtrar_evals_no_crecientes=False):
        """
        filtrar_evals_no_crecientes: si True, descarta entradas cuyo contador de evaluaciones no supera la última registrada (necesario para DE/SHADE).
        """
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
            "evals",
            "generacion",
            "tam_poblacion",
            "min/mejor_hasta_ahora",
            "promedio",
            "desv_std",
            "mediana",
            "max",
            "tiempo_s",
        ]
        self._diversidad_por_generacion = {}
        self._rangos_generacion = {}
        self._filtrar_evals_no_crecientes = bool(filtrar_evals_no_crecientes)
        self._ultima_eval_registrada = None
        

    # diversidad

    def _diversidad_dist_euclidea(self, poblacion):
        """
        Diversidad como distancia media euclidea de cada individuo al centroide. Mide como de lejos estan los individuos del centro de la poblacion (centroide).

        poblacion: array (n, dim) con los vectores de decisión de la población.
        """
        centroide = np.mean(poblacion, axis=0)
        dists = np.linalg.norm(poblacion - centroide, axis=1)
        return float(np.mean(dists))

    def _diversidad_dist_euclidea_normalizada(self, diversidad, dimension):
        """
        Diversidad euclidea normalizada por el número de dimensiones.

        diversidad: valor bruto devuelto por _diversidad_dist_euclidea.
        dimension: número de variables del problema.

        Divide la diversidad bruta entre dimension para hacerla comparable entre
        distintas configuraciones del problema.
        """
        dimension = int(dimension)
        if dimension <= 0:
            return float("nan")
        return float(diversidad) / dimension

    def registrar(self, generacion, fitness=None, evaluaciones=None, tiempo_s=None, tam_poblacion=None, poblacion=None, sobrescribir_ultima=False):
        """
        Registra una entrada en el logbook para la generación actual.

        generacion: índice de la generación.
        fitness: vector de fitness de la población (requerido).
        evaluaciones: número total de evaluaciones de la función objetivo (requerido).
        tiempo_s: tiempo transcurrido desde el inicio del experimento (requerido).
        tam_poblacion: si se omite, se deriva del tamaño del vector fitness.
        poblacion: array (n, dim) opcional para calcular diversidad euclidea.
        sobrescribir_ultima: si True, elimina la última entrada antes de añadir ésta.
        """

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
                self._ultima_eval_registrada = int(self.logbook[-1]["evals"])
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
            evals = int(evaluaciones),
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

    def obtener_logbook(self):
        """Devuelve el historial completo como lista de dicts."""
        return [dict(e) for e in self.logbook]

    def obtener_diversidad_por_generacion(self):
        """Devuelve el historial de diversidad calculado, ordenado por generación."""
        return dict(sorted(self._diversidad_por_generacion.items(), key=lambda kv: kv[0]))

    def anotar_rangos_generacion(self, rangos_generacion):
        """
        Almacena los rangos de eval_id por generación para el CSV.

        rangos_generacion: dict {generacion: {eval_id_inicio, eval_id_fin}} devuelto por SurrogateDataset.
        """
        self._rangos_generacion = dict(sorted(dict(rangos_generacion).items(), key=lambda kv: kv[0]))

    def anotar_diversidad_generacion(self, generacion, payload):
        """
        Añade la entrada de diversidad para una generación concreta.

        generacion: índice de la generación.
        payload: dict con claves div_dist_euclidea y/o div_dist_euclidea_normalizada.
        """
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

    def obtener_resumen_final(self):
        """
        Devuelve un dict con estadísticas de la última generación y totales acumulados.
        """
        historial = self.obtener_logbook()
        if len(historial) == 0:
            return {
                "generaciones_registradas": 0,
                "evaluaciones_totales": 0,
            }

        ultimo = historial[-1] # ultima generacion de pob
        resumen = {
            "generaciones_registradas": int(len(historial)),
            "evaluaciones_totales": int(ultimo["evals"]),
            "min_final": float(ultimo["min/mejor_hasta_ahora"]),
            "promedio_final": float(ultimo["promedio"]),
            "desv_std_final": float(ultimo["desv_std"]),
            "mediana_final": float(ultimo["mediana"]),
            "max_final": float(ultimo["max"]),
            "tiempo_total_s": float(ultimo["tiempo_s"]),
        }
        return resumen

def guardar_metricas_deap(recolector, ruta_base, metadata=None):
    """
    El historial se recoge como CSV y el resumen como JSON.

    recolector: instancia de RecolectorMetricasDEAP con la ejecución completada.
    ruta_base: directorio de salida.
    metadata: dict opcional incluido en el JSON de resumen.

    Retorna un dict con las rutas a los artefactos generados.
    """
    base = Path(ruta_base)
    base.mkdir(parents=True, exist_ok=True)

    historial = recolector.obtener_logbook()
    resumen = recolector.obtener_resumen_final()
    metadata_limpia = {k: v for k, v in metadata.items() if v is not None} if metadata else {}
    payload_resumen = {"metadata": metadata_limpia, "resumen": resumen}

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
        rango = recolector._rangos_generacion.get(gen, {})
        div = recolector._diversidad_por_generacion.get(gen, {})
        fila["eval_id_inicio"] = rango.get("eval_id_inicio")
        fila["eval_id_fin"] = rango.get("eval_id_fin")
        fila["div_dist_euclidea"] = div.get("div_dist_euclidea")
        fila["div_dist_euclidea_normalizada"] = div.get("div_dist_euclidea_normalizada")
        historial_enriquecido.append(fila)

    fieldnames = list(recolector.logbook.header) + [
        "eval_id_inicio", "eval_id_fin",
        "div_dist_euclidea", "div_dist_euclidea_normalizada",
    ]
    escribir_csv_dicts(ruta_logbook_csv, historial_enriquecido, fieldnames=fieldnames)
    escribir_json(ruta_resumen_json, payload_resumen)

    return {
        "resultados_csv": str(ruta_logbook_csv),
        "resumen_json": str(ruta_resumen_json),
    }
