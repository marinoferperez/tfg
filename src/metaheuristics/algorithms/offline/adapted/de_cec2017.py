"""
Wrapper de DE para el benchmark CEC2017.

Encapsula la configuración del experimento y delega la ejecución en
DifferentialEvolution, que usa internamente la implementación PYADE.
"""

import time
from pathlib import Path

from src.metaheuristics.algorithms.offline.de import DifferentialEvolution
from src.metaheuristics.metrics.elitist_restart import (
    construir_metadata_reinicios,
)
from src.benchmark.cec2017_problem import CEC2017Problem, MAX_EVALS_POR_DIM
from src.utils.experiment_io import guardar_reinicios_elitistas_csv
from src.metaheuristics.metrics.metrics_callback import CallbackMetricasDE
from src.metaheuristics.metrics.surrogate_dataset import SurrogateDataset, guardar_dataset_hdf5

class DifferentialEvolutionCEC2017:
    """Wrapper de DE para ejecutar una función CEC2017."""

    def __init__(self, **de_kwargs):
        """de_kwargs: argumentos para DifferentialEvolution (tam_poblacion, max_evals, …)."""
        self.de = DifferentialEvolution(**de_kwargs)

    def optimize(self, funcid, dim, seed=42, lib_path=None, algname="de", registrar_metricas=False, ruta_metricas=None, run_id=None, cec_workdir=None, guardar_reinicios_detalle=False):
        """
        Ejecuta DE sobre la función CEC2017 indicada.

        funcid: índice de la función CEC2017, en [1, 30].
        dim: dimensionalidad del problema.
        seed: semilla del generador aleatorio.
        lib_path: ruta opcional a la librería compilada de CEC2017.
        algname: etiqueta para la salida de cec2017real.
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
        self.de.seed = seed

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
                # el recolector acumula logbook por generación; el callback lo alimenta desde DE
                recolector = RecolectorMetricasDEAP(filtrar_evals_no_crecientes=True)
                tiempo_inicio = time.perf_counter()
                # el dataset recoge cada evaluación para entrenar el subrogado offline
                dataset = SurrogateDataset(
                    algoritmo="de",
                    problema="cec2017",
                    seed=seed,
                    run_info={"funcid": int(funcid), "dim": int(dim)},
                )
                callback_metricas = CallbackMetricasDE(
                    recolector,
                    tiempo_inicio,
                    lambda: self.de.evals,
                    en_generacion=lambda g: setattr(self.de, "_generacion_actual", int(g) + 1),
                    offset_current_generation=1,
                    restart_manager=self.de._aplicar_reinicio,
                )

            # ejecución del algoritmo
            mejor_sol, mejor_fitness = self.de.optimize(
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

                # config expone los parámetros reales que PYADE usó (tam_poblacion, f, cr…)
                config = getattr(callback_metricas, "config", None) or {}

                evals_objetivo = config.get("max_evals")
                if evals_objetivo is None:
                    evals_objetivo = int(self.de.max_evals) if self.de.max_evals is not None else int(MAX_EVALS_POR_DIM * dim)

                evals_reales = int(self.de.evals)
                # PYADE puede evaluar más allá del presupuesto por trabajar en bloques de generación
                evals_fuera_presupuesto = int(max(0, evals_reales - evals_objetivo))
                hubo_fuera_presupuesto = bool(evals_fuera_presupuesto > 0)

                if ruta_metricas is not None:
                    # run_id identifica de forma única esta ejecución en el sistema de archivos
                    if run_id is None:
                        run_id = f"de_cec2017_f{int(funcid)}_d{int(dim)}_s{seed}"
                    ruta_base = Path(ruta_metricas) / run_id
                    if dataset is not None:
                        rangos_generacion = dataset.obtener_rangos_generacion()
                        recolector.anotar_rangos_generacion(rangos_generacion)
                        # completar gen=0 en el recolector antes de anotar el dataset,
                        # ya que el buffer no tiene población y no pudo calcularla
                        diversidad_por_generacion = recolector.obtener_diversidad_por_generacion()
                        if 0 not in diversidad_por_generacion:
                            rango_gen0 = rangos_generacion.get(0)
                            if rango_gen0 is not None:
                                diversidad_gen0 = dataset.calcular_diversidad_rango(
                                    rango_gen0["eval_id_inicio"],
                                    rango_gen0["eval_id_fin"],
                                )
                                if diversidad_gen0 is not None:
                                    recolector.anotar_diversidad_generacion(0, diversidad_gen0)
                        # anotar el dataset con la diversidad completa (gen=0 ya incluida)
                        dataset.anotar_diversidad_por_generacion(recolector.obtener_diversidad_por_generacion())
                    # metadata_reinicios agrega campos de reinicio al JSON de configuración
                    metadata_reinicios = construir_metadata_reinicios(
                        self.de.eventos_reinicio,
                        self.de.reinicio_ratio,
                        self.de.reinicio,
                    )
                    ficheros_metricas = guardar_metricas_deap(recolector,
                        ruta_base=ruta_base,
                        metadata={
                            "algoritmo": "de",
                            "problema": "cec2017",
                            "funcid": int(funcid),
                            "dim": int(dim),
                            "seed": int(seed),
                            "tam_poblacion": config.get("population_size"),
                            "max_evals_objetivo": evals_objetivo,
                            "f": config.get("f"),
                            "cr": config.get("cr"),
                            "cross": config.get("cross"),
                            "evals_reales": evals_reales,
                            "evals_fuera_presupuesto": evals_fuera_presupuesto,
                            "hubo_fuera_presupuesto": hubo_fuera_presupuesto,
                            **metadata_reinicios,
                        },
                    )
                    ruta_reinicios_csv = None
                    if guardar_reinicios_detalle:
                        # CSV opcional con el detalle de cada evento de reinicio elitista
                        ruta_reinicios_csv = guardar_reinicios_elitistas_csv(
                            ruta_base,
                            self.de.eventos_reinicio,
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
                    run_id = f"de_cec2017_f{int(funcid)}_d{int(dim)}_s{seed}"
                ruta_base = Path(ruta_metricas) / run_id
                ruta_base.mkdir(parents=True, exist_ok=True)
                ruta_reinicios_csv = guardar_reinicios_elitistas_csv(
                    ruta_base,
                    self.de.eventos_reinicio,
                )
                if ruta_reinicios_csv is not None:
                    resultado["ruta_reinicios_elitistas_csv"] = ruta_reinicios_csv

            # los eventos de reinicio se devuelven siempre para facilitar el análisis
            resultado["reinicios"] = list(self.de.eventos_reinicio)

            return resultado
        finally:
            problema.exit_workdir()
