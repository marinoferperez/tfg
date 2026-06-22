"""
Wrapper de SHADE para el benchmark CEC2017.

Encapsula la configuración del experimento y delega la ejecución en SHADE,
variante adaptativa de DE con memoria histórica de F y CR.
"""

import time
from pathlib import Path

import numpy as np

from src.metaheuristics.algorithms.offline.shade import SHADE
from src.metaheuristics.metrics.elitist_restart import (
    construir_metadata_reinicios,
)
from src.benchmark.cec2017_problem import CEC2017Problem, MAX_EVALS_POR_DIM
from src.utils.experiment_io import guardar_reinicios_elitistas_csv, guardar_dataset_hdf5
from src.metaheuristics.metrics.metrics_callback import CallbackMetricasDE
from src.metaheuristics.metrics.surrogate_dataset import SurrogateDataset


class SHADECEC2017:
    """Wrapper de SHADE para ejecutar una función CEC2017."""

    def __init__(self, **shade_kwargs):
        """shade_kwargs: argumentos para SHADE (tam_poblacion, max_evals, memory_size, …)."""
        self.shade = SHADE(**shade_kwargs)

    def optimize(self, funcid, dim, seed=42, lib_path=None, algname="shade", registrar_metricas=False, ruta_metricas=None, run_id=None, cec_workdir=None, guardar_reinicios_detalle=False):
        """
        Ejecuta SHADE sobre la función CEC2017 indicada.

        funcid: índice de la función CEC2017, en [1, 30].
        dim: dimensionalidad del problema.
        seed: semilla del generador aleatorio.
        lib_path: ruta opcional a la librería compilada de CEC2017.
        algname: etiqueta para la salida de cec2017real.
        registrar_metricas: si True, genera CSV/JSON de métricas y dataset.
        ruta_metricas: directorio raíz donde guardar los ficheros.
        run_id: nombre del subdirectorio de ficheros. Si es None, se genera automáticamente.
        cec_workdir: directorio de trabajo para cec2017real.
        guardar_reinicios_detalle: si True, guarda un CSV con el detalle de cada reinicio elitista.

        Retorna un dict con mejor_sol, mejor_fitness, mejor_error y, si
        registrar_metricas=True, las rutas a los ficheros generados.
        """
        seed = int(seed)
        self.shade.seed = seed

        # construcción del problema
        problema = CEC2017Problem(funcid=funcid, dim=dim, algname=algname, lib_path=lib_path, seed=seed, workdir=cec_workdir)
        
        problema.enter_workdir()
        try:
            problema.prepare_run()

            # registro de métricas
            recolector = None
            callback_metricas = None
            dataset = None

            if registrar_metricas:
                from src.metaheuristics.metrics.deap_metrics import RecolectorMetricasDEAP, guardar_metricas_deap

                recolector = RecolectorMetricasDEAP(filtrar_evals_no_crecientes=True)
                tiempo_inicio = time.perf_counter()
                # el dataset recoge cada evaluación para entrenar el subrogado offline
                dataset = SurrogateDataset(algoritmo="shade", problema="cec2017", run_info={"funcid": int(funcid), "dim": int(dim)})
                callback_metricas = CallbackMetricasDE(recolector, tiempo_inicio, lambda: self.shade.evals, en_generacion=lambda g: setattr(self.shade, "_generacion_actual", int(g) + 1), offset_current_generation=1, restart_manager=self.shade._aplicar_reinicio)

            # ejecución del algoritmo
            mejor_sol, mejor_fitness = self.shade.optimize(limites=problema.get_bounds(), problema=problema, callback_metricas=callback_metricas, dataset=dataset)

            mejor_error = problema.cec_error(mejor_fitness)

            # resultado mínimo independientemente de registrar_metricas
            resultado = {
                "mejor_sol": mejor_sol,
                "mejor_fitness": float(mejor_fitness),
                "mejor_error": mejor_error,
            }

            # postprocesado de métricas
            if registrar_metricas:
                metricas_resumen = recolector.obtener_resumen_final()
                # se añaden los valores finales para tenerlos en el JSON de resumen
                metricas_resumen["mejor_fitness"] = float(mejor_fitness)
                metricas_resumen["mejor_error"] = float(mejor_error)

                metricas_logbook = recolector.obtener_logbook()
                resultado["metricas_logbook"] = metricas_logbook
                resultado["metricas_resumen"] = metricas_resumen

                # config recoge los parámetros reales que PYADE usó (tam_poblacion, memory_size…)
                config = getattr(callback_metricas, "config", None) or {}

                evals_objetivo = config.get("max_evals")
                if evals_objetivo is None:
                    evals_objetivo = int(self.shade.max_evals) if self.shade.max_evals is not None else int(MAX_EVALS_POR_DIM * dim)

                evals_reales = int(self.shade.evals)
                # PYADE puede evaluar más allá del presupuesto por trabajar en bloques de generación
                evals_fuera_presupuesto = int(max(0, evals_reales - evals_objetivo))
                hubo_fuera_presupuesto = bool(evals_fuera_presupuesto > 0)

                if ruta_metricas is not None:
                    # run_id identifica esta ejecución en el arch
                    if run_id is None:
                        run_id = f"shade_cec2017_f{int(funcid)}_d{int(dim)}_s{seed}"
                        
                    ruta_base = Path(ruta_metricas) / run_id
                    
                    if dataset is not None and 0 not in recolector.obtener_diversidad_por_generacion():
                        population_size = (config.get("population_size") or 0)
                        if population_size > 0:
                            from src.metaheuristics.metrics.deap_metrics import calcular_diversidad_euclidea
                            puntos_gen0 = [np.asarray(f["x"], dtype=float) for f in dataset.filas[:population_size] if "x" in f]
                            diversidad_gen0 = calcular_diversidad_euclidea(puntos_gen0)
                            if diversidad_gen0 is not None:
                                recolector.anotar_diversidad_generacion(0, diversidad_gen0)

                    # metadata_reinicios agrega campos de reinicio al JSON de configuración
                    metadata_reinicios = construir_metadata_reinicios(self.shade.eventos_reinicio, self.shade.reinicio_ratio, self.shade.reinicio)
                    ficheros_metricas = guardar_metricas_deap(recolector, ruta_base=ruta_base, metadata={"algoritmo": "shade", "problema": "cec2017", "funcid": int(funcid), "dim": int(dim), "seed": int(seed), "tam_poblacion": config.get("population_size"), "max_evals_objetivo": evals_objetivo, "memory_size": config.get("memory_size"), "evals_reales": evals_reales, "evals_fuera_presupuesto": evals_fuera_presupuesto, "hubo_fuera_presupuesto": hubo_fuera_presupuesto, **metadata_reinicios})
                    ruta_reinicios_csv = None
                    
                    if guardar_reinicios_detalle:
                        # CSV opcional con el detalle de cada evento de reinicio elitista
                        ruta_reinicios_csv = guardar_reinicios_elitistas_csv(ruta_base, self.shade.eventos_reinicio)
                        
                    ficheros_dataset = guardar_dataset_hdf5(dataset, ruta_base) if dataset is not None else None
                    resultado["ruta_metricas"] = str(ruta_base)
                    resultado["ficheros_metricas"] = ficheros_metricas
                    resultado["ficheros_dataset"] = ficheros_dataset
                    
                    if ruta_reinicios_csv is not None:
                        resultado["ruta_reinicios_elitistas_csv"] = ruta_reinicios_csv

            # los eventos de reinicio se devuelven siempre para facilitar el análisis
            resultado["reinicios"] = list(self.shade.eventos_reinicio)

            return resultado
        finally:
            problema.exit_workdir()
