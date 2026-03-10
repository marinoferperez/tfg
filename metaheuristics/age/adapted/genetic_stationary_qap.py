import numpy as np
import time
from pathlib import Path

from metaheuristics.age.genetic_stationary_combinatorial import GeneticAlgorithmCombinatorio
from metaheuristics.problems.qap_problem import QAPProblem
from metaheuristics.metrics import RecolectorMetricasDEAP, CallbackMetricasAGE, SurrogateDataset

class GeneticStationaryQAP:
    def __init__(self, **age_kwargs):
        self.age = GeneticAlgorithmCombinatorio(**age_kwargs)

    def optimize(self, qap_path = None, mat_flujo = None, mat_dist = None, seed = 42, registrar_metricas = False, ruta_metricas = None, run_id = None):
        seed = int(seed)
        self.age.rng = np.random.default_rng(seed)

        # ----------------------------
        # construccion del problema
        # ----------------------------
        # se carga el problema QAP
        # o
        # se define manualmente con la matriz de flujo y distancias
        if qap_path is not None:
            problema = QAPProblem.from_qaplib(path = qap_path, seed = seed, tipo_algoritmo = "combinatorio")
            instancia = Path(qap_path).stem
        else:
            if mat_flujo is None or mat_dist is None:
                raise ValueError("Debes pasar qap_path o (mat_flujo, mat_dist)")

            problema = QAPProblem(
                mat_flujo = mat_flujo,
                mat_distancias = mat_dist,
                seed = seed,
                tipo_algoritmo = "combinatorio",
            )
            instancia = "custom"

        # ----------------------------
        # registro de métricas (DEAP)
        # ----------------------------
        recolector = None
        callback_metricas = None

        #####

        dataset = None

        if registrar_metricas:
            recolector = RecolectorMetricasDEAP()
            tiempo_inicio = time.perf_counter()
            callback_metricas = CallbackMetricasAGE(recolector, tiempo_inicio)
            dataset = SurrogateDataset(
                algoritmo = "age",
                problema = "qap",
                seed = seed,
                run_info = {"instancia": str(instancia)},
            )

        # if registrar_metricas:
        #     recolector = RecolectorMetricasDEAP()
        #     tiempo_inicio = time.perf_counter()
        #     callback_metricas = CallbackMetricasAGE(recolector, tiempo_inicio)


        # ----------------------------
        # ejecucion del algoritmo AGE
        # ----------------------------
        mejor_sol, mejor_fitness = self.age.optimize(problem = problema, callback_metricas = callback_metricas, dataset = dataset)
        resultado = {
            "mejor_sol": mejor_sol,
            "mejor_fitness": float(mejor_fitness)
        }

        # ----------------------------
        # postprocesado de métricas
        # ----------------------------
        if registrar_metricas:
            metricas_logbook = recolector.obtener_logbook()
            metricas_resumen = recolector.obtener_resumen_final()

            metricas_resumen["mejor_fitness"] = float(mejor_fitness)

            resultado["metricas_logbook"] = metricas_logbook
            resultado["metricas_resumen"] = metricas_resumen

            if ruta_metricas is not None:
                if run_id is None:
                    run_id = (
                        f"age_qap_{instancia}_"
                        f"n{int(problema.get_size())}_"
                        f"s{seed}"
                    )
                ruta_base = Path(ruta_metricas) / run_id
                ficheros_metricas = recolector.guardar_csv_json(
                    ruta_base = ruta_base,
                    metadata = {
                        "algoritmo": "age",
                        "problema": "qap",
                        "instancia": str(instancia),
                        "n": int(problema.get_size()),
                        "k_pares_hamming": int(getattr(recolector, "_k_pares_hamming", 200)),
                        "hamming_fuente": "permutacion",
                        "hamming_normalizada_01": True,
                        "tam_poblacion": int(self.age.tam_poblacion),
                        "prob_cruce": float(self.age.prob_cruce),
                        "prob_mutacion": float(self.age.prob_mutacion),
                        "tam_torneo": int(self.age.tam_torneo),
                        "max_evals": int(self.age.max_evals) if self.age.max_evals else None,
                        "seed": int(seed),
                    },
                )
                # resultado["ruta_metricas"] = str(ruta_base)
                # resultado["ficheros_metricas"] = ficheros_metricas
                if dataset is not None:
                    dataset.anotar_diversidad_por_generacion(recolector.obtener_diversidad_por_generacion())
                ficheros_dataset = dataset.guardar_csv_json(ruta_base) if dataset is not None else None
                resultado["ruta_metricas"] = str(ruta_base)
                resultado["ficheros_metricas"] = ficheros_metricas
                resultado["ficheros_dataset"] = ficheros_dataset

        return resultado
