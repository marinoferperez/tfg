"""
Wrapper de AGE para el benchmark CEC2017.

Encapsula la configuración del experimento (semilla, directorio de trabajo,
registro de métricas y dataset de subrogado) y expone una interfaz de un solo
método: optimize(funcid, dim, ...).
"""

import numpy as np
import time
from pathlib import Path

from src.metaheuristics.algorithms.offline.age import GeneticoEstacionario
from src.metaheuristics.metrics.elitist_restart import (
    construir_metadata_reinicios,
)
from src.benchmark.cec2017_problem import CEC2017Problem
from src.utils.experiment_io import guardar_reinicios_elitistas_csv
from src.metaheuristics.metrics.metrics_callback import CallbackMetricasAGE
from src.metaheuristics.metrics.surrogate_dataset import SurrogateDataset, guardar_dataset_hdf5

class GeneticStationaryCEC2017:
    """Wrapper de AGE para ejecutar una función CEC2017."""

    def __init__(self, **age_kwargs):
        """age_kwargs: argumentos para GeneticoEstacionario (tam_poblacion, max_evals, …)."""
        self.age = GeneticoEstacionario(**age_kwargs)

    def optimize(self, funcid, dim, seed=42, algname="age", lib_path=None, registrar_metricas=False, ruta_metricas=None, run_id=None, cec_workdir=None, guardar_reinicios_detalle=False):
        """
        Ejecuta AGE sobre la función CEC2017 indicada.

        funcid: índice de la función CEC2017, en [1, 30].
        dim: dimensionalidad del problema.
        seed: semilla del generador aleatorio.
        algname: etiqueta para la salida de cec2017real.
        lib_path: ruta opcional a la librería compilada de CEC2017.
        registrar_metricas: si True, genera CSV/JSON de métricas y dataset.
        ruta_metricas: directorio raíz donde guardar los artefactos.
        run_id: nombre del subdirectorio de artefactos. Si es None, se genera automáticamente.
        cec_workdir: directorio de trabajo para cec2017real.
        guardar_reinicios_detalle: si True, guarda un CSV con el detalle de cada reinicio elitista.

        Retorna un dict con mejor_sol, mejor_fitness, mejor_error y, si
        registrar_metricas=True, las rutas a los artefactos generados.
        """
        # la semilla se fuerza a int para que numpy no rechace tipos flotantes
        seed = int(seed)
        self.age.rng = np.random.default_rng(seed)

        # construcción del problema
        problema = CEC2017Problem(
            funcid=funcid,
            dim=dim,
            algname=algname,
            lib_path=lib_path,
            seed=seed,
            workdir=cec_workdir,
        )
        # enter_workdir cambia al directorio que necesita la librería C de CEC2017
        problema.enter_workdir()
        try:
            # prepare_run inicializa el estado interno de cec2017real para esta función
            problema.prepare_run()

            # registro de métricas
            recolector = None
            callback_metricas = None
            dataset = None

            if registrar_metricas:
                from src.metaheuristics.metrics.deap_metrics import RecolectorMetricasDEAP, guardar_metricas_deap
                # el recolector acumula logbook por generación; el callback lo alimenta desde AGE
                recolector = RecolectorMetricasDEAP()
                tiempo_inicio = time.perf_counter()
                callback_metricas = CallbackMetricasAGE(recolector, tiempo_inicio)
                # el dataset recoge cada evaluación para entrenar el subrogado offline
                dataset = SurrogateDataset(
                    algoritmo="age",
                    problema="cec2017",
                    seed=seed,
                    run_info={"funcid": int(funcid), "dim": int(dim)},
                )

            # ejecución del algoritmo
            mejor_sol, mejor_fitness = self.age.optimize(
                limites=problema.get_bounds(),
                problema=problema,
                callback_metricas=callback_metricas,
                dataset=dataset,
            )

            # mejor_error es la distancia al óptimo conocido de CEC2017 (f - f*)
            mejor_error = problema.cec_error(mejor_fitness)

            # resultado mínimo siempre presente, independientemente de registrar_metricas
            resultado = {
                "mejor_sol": mejor_sol,
                "mejor_fitness": float(mejor_fitness),
                "mejor_error": mejor_error,
            }

            # postprocesado de métricas
            if registrar_metricas:
                metricas_resumen = recolector.obtener_resumen_final()
                # se inyectan los valores finales para tenerlos en el JSON de resumen
                metricas_resumen["mejor_fitness"] = float(mejor_fitness)
                metricas_resumen["mejor_error"] = float(mejor_error)

                metricas_logbook = recolector.obtener_logbook()
                resultado["metricas_logbook"] = metricas_logbook
                resultado["metricas_resumen"] = metricas_resumen

                if ruta_metricas is not None:
                    # run_id identifica de forma única esta ejecución en el sistema de archivos
                    if run_id is None:
                        run_id = f"age_cec2017_f{int(funcid)}_d{int(dim)}_s{seed}"
                    ruta_base = Path(ruta_metricas) / run_id
                    if dataset is not None:
                        # se cruzan los rangos de evaluación por generación entre recolector y dataset
                        dataset.anotar_diversidad_por_generacion(recolector.obtener_diversidad_por_generacion())
                        recolector.anotar_rangos_generacion(dataset.obtener_rangos_generacion())
                    # metadata_reinicios agrega campos de reinicio al JSON de configuración
                    metadata_reinicios = construir_metadata_reinicios(
                        self.age.eventos_reinicio,
                        self.age.reinicio_ratio,
                        self.age.reinicio,
                    )
                    ficheros_metricas = guardar_metricas_deap(recolector,
                        ruta_base=ruta_base,
                        metadata={
                            "algoritmo": "age",
                            "problema": "cec2017",
                            "funcid": int(funcid),
                            "dim": int(dim),
                            "seed": int(seed),
                            "tam_poblacion": int(self.age.tam_poblacion),
                            "prob_cruce": float(self.age.prob_cruce),
                            "prob_mutacion": float(self.age.prob_mutacion),
                            "tam_torneo": int(self.age.tam_torneo),
                            "max_evals": int(self.age.max_evals) if self.age.max_evals else None,
                            "sigma": float(self.age.sigma),
                            "alpha": float(self.age.alpha),
                            **metadata_reinicios,
                        },
                    )
                    ruta_reinicios_csv = None
                    if guardar_reinicios_detalle:
                        # CSV opcional con el detalle de cada evento de reinicio elitista
                        ruta_reinicios_csv = guardar_reinicios_elitistas_csv(
                            ruta_base,
                            self.age.eventos_reinicio,
                        )
                    ficheros_dataset = guardar_dataset_hdf5(dataset, ruta_base) if dataset is not None else None
                    resultado["ruta_metricas"] = str(ruta_base)
                    resultado["ficheros_metricas"] = ficheros_metricas
                    resultado["ficheros_dataset"] = ficheros_dataset
                    if ruta_reinicios_csv is not None:
                        resultado["ruta_reinicios_elitistas_csv"] = ruta_reinicios_csv

            # si se pidió el CSV de reinicios pero no se guardaron métricas, se hace aquí
            if (
                guardar_reinicios_detalle
                and ruta_metricas is not None
                and "ruta_reinicios_elitistas_csv" not in resultado
            ):
                if run_id is None:
                    run_id = f"age_cec2017_f{int(funcid)}_d{int(dim)}_s{seed}"
                ruta_base = Path(ruta_metricas) / run_id
                ruta_base.mkdir(parents=True, exist_ok=True)
                ruta_reinicios_csv = guardar_reinicios_elitistas_csv(
                    ruta_base,
                    self.age.eventos_reinicio,
                )
                if ruta_reinicios_csv is not None:
                    resultado["ruta_reinicios_elitistas_csv"] = ruta_reinicios_csv

            # los eventos de reinicio se devuelven siempre para facilitar el análisis
            resultado["reinicios"] = list(self.age.eventos_reinicio)

            return resultado
        finally:
            problema.exit_workdir()
