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

        Retorna un dict con mejor_sol, mejor_fitness, mejor_error y, si
        registrar_metricas=True, las rutas a los artefactos generados.
        """
        seed = int(seed)
        self.age.rng = np.random.default_rng(seed)

        # ----------------------------
        # construccion del problema
        # ----------------------------
        # se carga el problema CEC (funcid, dim)
        problema = CEC2017Problem(
            funcid = funcid,
            dim = dim,
            algname = algname,
            lib_path = lib_path,
            seed = seed,
            workdir = cec_workdir,
        )
        problema.enter_workdir()
        try:
            problema.prepare_run()

            # ----------------------------
            # registro de métricas (CEC2017)
            # ----------------------------
            recolector = None
            callback_metricas = None

            #####

            dataset = None
            if registrar_metricas:
                from src.metaheuristics.metrics.deap_metrics import RecolectorMetricasDEAP, guardar_metricas_deap
                recolector = RecolectorMetricasDEAP()
                tiempo_inicio = time.perf_counter()
                callback_metricas = CallbackMetricasAGE(recolector, tiempo_inicio)
                dataset = SurrogateDataset(
                    algoritmo = "age",
                    problema = "cec2017",
                    seed = seed,
                    run_info = {"funcid": int(funcid), "dim": int(dim)},
                )

        # ----------------------------
        # ejecucion del algoritmo AGE
        # ----------------------------
            mejor_sol, mejor_fitness = self.age.optimize(
                limites = problema.get_bounds(),
                problem = problema,
                callback_metricas = callback_metricas,
                dataset = dataset
            )

            mejor_error = problema.cec_error(mejor_fitness)

            resultado = {
                "mejor_sol": mejor_sol,
                "mejor_fitness": float(mejor_fitness),
                "mejor_error": mejor_error,
            }

            # ----------------------------
            # postprocesado de métricas
            # ----------------------------
            if registrar_metricas:
                metricas_resumen = recolector.obtener_resumen_final()
                metricas_resumen["mejor_fitness"] = float(mejor_fitness)
                metricas_resumen["mejor_error"] = float(mejor_error)

                metricas_logbook = recolector.obtener_logbook()
                resultado["metricas_logbook"] = metricas_logbook
                resultado["metricas_resumen"] = metricas_resumen

                if ruta_metricas is not None:
                    if run_id is None:
                        run_id = f"age_cec2017_f{int(funcid)}_d{int(dim)}_s{seed}"
                    ruta_base = Path(ruta_metricas) / run_id
                    if dataset is not None:
                        dataset.anotar_diversidad_por_generacion(recolector.obtener_diversidad_por_generacion())
                        recolector.anotar_rangos_generacion(dataset.obtener_rangos_generacion())
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
            resultado["reinicios"] = list(self.age.eventos_reinicio)

            return resultado
        finally:
            problema.exit_workdir()
