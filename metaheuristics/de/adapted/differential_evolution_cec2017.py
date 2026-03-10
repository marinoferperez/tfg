import numpy as np
import time
from pathlib import Path

from metaheuristics.de.differential_evolution import DifferentialEvolution
from metaheuristics.problems.cec2017_problem import CEC2017Problem
from metaheuristics.metrics import RecolectorMetricasDEAP, CallbackMetricasDE, SurrogateDataset

class DifferentialEvolutionCEC2017:
    def __init__(self, **de_kwargs):
        self.de = DifferentialEvolution(**de_kwargs)

    def optimize(self, funcid, dim, seed = 42, lib_path = None, algname = "de_cec", registrar_metricas = False, ruta_metricas = None, run_id = None):
        seed = int(seed)
        self.de.seed = seed

        # ----------------------------
        # construccion del problema
        # ----------------------------
        # se carga el problema CEC (funcid, dim)
        problema = CEC2017Problem(funcid = funcid, dim = dim, algname = algname, lib_path = lib_path, seed = seed)
        problema.prepare_run()

        # ----------------------------
        # registro de métricas (CEC2017)
        # ----------------------------
        recolector = None
        callback_metricas = None

        dataset = None
        if registrar_metricas:
            recolector = RecolectorMetricasDEAP()
            tiempo_inicio = time.perf_counter()
            dataset = SurrogateDataset(
                algoritmo = "de",
                problema = "cec2017",
                seed = seed,
                run_info = {"funcid": int(funcid), "dim": int(dim)},
            )
            callback_metricas = CallbackMetricasDE(recolector, 
            tiempo_inicio, 
            lambda: self.de.evals,
            en_generacion = lambda g: setattr(self.de, "_generacion_actual", int(g))
            )


        # if registrar_metricas:
        #     recolector = RecolectorMetricasDEAP()
        #     tiempo_inicio = time.perf_counter()
        #     callback_metricas = CallbackMetricasDE(recolector, tiempo_inicio, lambda: self.de.evals)

        # ----------------------------
        # ejecucion del algoritmo AGE
        # ----------------------------
        mejor_sol, mejor_fitness = self.de.optimize(
            limites = problema.get_bounds(),
            problem = problema,
            callback_metricas = callback_metricas,
            dataset = dataset,
            perm_decodificador = None
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

            # metricas
            config = getattr(callback_metricas, "config", None) or {}

            evals_objetivo = config.get("max_evals")
            if evals_objetivo is None:
                evals_objetivo = int(self.de.max_evals) if self.de.max_evals is not None else int(10000 * dim)

            # tam_poblacion = int(self.de.tam_poblacion) if self.de.tam_poblacion is not None else int(10 * dim)
            # evals_objetivo = int(self.de.max_evals) if self.de.max_evals is not None else int(10000 * dim)
            # f = float(self.de.f) if self.de.f is not None else 0.5
            # cr = float(self.de.cr) if self.de.cr is not None else 0.9
            # cross = str(self.de.metodo_cruce) if self.de.metodo_cruce is not None else "bin"

            evals_reales = int(self.de.evals) 
            evals_fuera_presupuesto = int(max(0, evals_reales - evals_objetivo))
            hubo_fuera_presupuesto = bool(evals_fuera_presupuesto > 0)

            if ruta_metricas is not None:
                if run_id is None:
                    run_id = f"de_cec2017_f{int(funcid)}_d{int(dim)}_s{seed}"
                ruta_base = Path(ruta_metricas) / run_id
                ficheros_metricas = recolector.guardar_csv_json(
                    ruta_base = ruta_base,
                    metadata = {
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
                    },
                )
                if dataset is not None:
                    dataset.anotar_diversidad_por_generacion(recolector.obtener_diversidad_por_generacion())
                ficheros_dataset = dataset.guardar_csv_json(ruta_base) if dataset is not None else None
                resultado["ruta_metricas"] = str(ruta_base)
                resultado["ficheros_metricas"] = ficheros_metricas
                resultado["ficheros_dataset"] = ficheros_dataset

        return resultado
