import numpy as np
import time
from pathlib import Path

from metaheuristics.shade.shade import SHADE
from metaheuristics.metrics.reinicio_elitista import (
    construir_metadata_reinicios,
    guardar_reinicios_elitistas_csv,
)
from metaheuristics.problems.cec2017_problem import CEC2017Problem
from metaheuristics.metrics import CallbackMetricasDE, SurrogateDataset


class SHADECEC2017:
    def __init__(self, **shade_kwargs):
        self.shade = SHADE(**shade_kwargs)

    def optimize(
        self,
        funcid,
        dim,
        seed=42,
        lib_path=None,
        algname="shade_cec",
        registrar_metricas=False,
        ruta_metricas=None,
        run_id=None,
        cec_workdir=None,
    ):
        seed = int(seed)
        self.shade.seed = seed

        # ----------------------------
        # construcción del problema
        # ----------------------------
        problema = CEC2017Problem(
            funcid=funcid,
            dim=dim,
            algname=algname,
            lib_path=lib_path,
            seed=seed,
            workdir=cec_workdir,
        )
        problema.enter_workdir()
        try:
            problema.prepare_run()

            # ----------------------------
            # registro de métricas
            # ----------------------------
            recolector = None
            callback_metricas = None
            dataset = None

            if registrar_metricas:
                from metaheuristics.metrics import RecolectorMetricasDEAP
                recolector = RecolectorMetricasDEAP(filtrar_evals_no_crecientes=True)
                tiempo_inicio = time.perf_counter()
                dataset = SurrogateDataset(
                    algoritmo="shade",
                    problema="cec2017",
                    seed=seed,
                    run_info={"funcid": int(funcid), "dim": int(dim)},
                )
                callback_metricas = CallbackMetricasDE(
                    recolector,
                    tiempo_inicio,
                    lambda: self.shade.evals,
                    en_generacion=lambda g: setattr(self.shade, "_generacion_actual", int(g) + 1),
                    offset_current_generation=1,
                    restart_manager=self.shade.aplicar_reinicio_elitista_desde_estado,
                )

            # ----------------------------
            # ejecución del algoritmo SHADE
            # ----------------------------
            mejor_sol, mejor_fitness = self.shade.optimize(
                limites=problema.get_bounds(),
                problem=problema,
                callback_metricas=callback_metricas,
                dataset=dataset,
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

                config = getattr(callback_metricas, "config", None) or {}

                evals_objetivo = config.get("max_evals")
                if evals_objetivo is None:
                    evals_objetivo = int(self.shade.max_evals) if self.shade.max_evals is not None else int(10000 * dim)

                evals_reales = int(self.shade.evals)
                evals_fuera_presupuesto = int(max(0, evals_reales - evals_objetivo))
                hubo_fuera_presupuesto = bool(evals_fuera_presupuesto > 0)

                if ruta_metricas is not None:
                    if run_id is None:
                        run_id = f"shade_cec2017_f{int(funcid)}_d{int(dim)}_s{seed}"
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
                    metadata_reinicios = construir_metadata_reinicios(
                        self.shade.eventos_reinicio_elitista,
                        self.shade.reinicio_elitista_ratio_estabilidad_diversidad,
                        self.shade.reinicio_elitista_ratio_paciencia,
                        self.shade.reinicio_elitista,
                    )
                    ficheros_metricas = recolector.guardar_csv_json(
                        ruta_base=ruta_base,
                        metadata={
                            "algoritmo": "shade",
                            "problema": "cec2017",
                            "funcid": int(funcid),
                            "dim": int(dim),
                            "seed": int(seed),
                            "tam_poblacion": config.get("population_size"),
                            "max_evals_objetivo": evals_objetivo,
                            "memory_size": config.get("memory_size"),
                            "evals_reales": evals_reales,
                            "evals_fuera_presupuesto": evals_fuera_presupuesto,
                            "hubo_fuera_presupuesto": hubo_fuera_presupuesto,
                            **metadata_reinicios,
                        },
                    )
                    ruta_reinicios_csv = guardar_reinicios_elitistas_csv(
                        ruta_base,
                        self.shade.eventos_reinicio_elitista,
                    )
                    ficheros_dataset = dataset.guardar_csv_json(ruta_base) if dataset is not None else None
                    resultado["ruta_metricas"] = str(ruta_base)
                    resultado["ficheros_metricas"] = ficheros_metricas
                    resultado["ficheros_dataset"] = ficheros_dataset
                    resultado["ruta_reinicios_elitistas_csv"] = ruta_reinicios_csv
            resultado["reinicios_elitistas"] = list(self.shade.eventos_reinicio_elitista)

            return resultado
        finally:
            problema.exit_workdir()
