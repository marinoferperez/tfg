import numpy as np
import time
from pathlib import Path

from metaheuristics.age.genetic_stationary_continuous import GeneticAlgorithmContinuo
from metaheuristics.problems.cec2017_problem import CEC2017Problem
from metaheuristics.metrics import RecolectorMetricasDEAP, CallbackMetricas

class GeneticStationaryCEC2017:
    def __init__(self, **age_kwargs):
        self.age = GeneticAlgorithmContinuo(**age_kwargs)

    def optimize(self, funcid, dim, seed = 42, algname = "age", lib_path=None, registrar_metricas = False, ruta_metricas = None, run_id = None):
        seed = int(seed)
        self.age.rng = np.random.default_rng(seed)

        # ----------------------------
        # construccion del problema
        # ----------------------------
        # se carga el problema CEC (funcid, dim)
        problema = CEC2017Problem(funcid=funcid, dim=dim, algname=algname, lib_path=lib_path, seed=seed)
        problema.prepare_run()

        # ----------------------------
        # registro de métricas (CEC2017)
        # ----------------------------
        recolector = None
        callback_metricas = None

        if registrar_metricas:
            recolector = RecolectorMetricasDEAP()
            tiempo_inicio = time.perf_counter()
            callback_metricas = CallbackMetricas(recolector, tiempo_inicio)

        # ----------------------------
        # ejecucion del algoritmo AGE
        # ----------------------------
        mejor_sol, mejor_fitness = self.age.optimize(
            limites = problema.get_bounds(),
            problem = problema,
            callback_metricas = callback_metricas
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
                ficheros_metricas = recolector.guardar_csv_json(
                    ruta_base=ruta_base,
                    metadata={
                        "algoritmo": "age",
                        "problema": "cec2017",
                        "funcid": int(funcid),
                        "dim": int(dim),
                        "seed": int(seed),
                        "algname": str(algname),
                        "tam_poblacion": int(self.age.tam_poblacion),
                        "prob_cruce": float(self.age.prob_cruce),
                        "prob_mutacion": float(self.age.prob_mutacion),
                        "tam_torneo": int(self.age.tam_torneo),
                        "max_evals": int(self.age.max_evals) if self.age.max_evals else None,
                        "sigma": float(self.age.sigma),
                        "alpha": float(self.age.alpha)
                    },
                )
                resultado["ruta_metricas"] = str(ruta_base)
                resultado["ficheros_metricas"] = ficheros_metricas

        return resultado
