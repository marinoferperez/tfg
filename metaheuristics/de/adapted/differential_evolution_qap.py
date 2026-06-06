import numpy as np
import time
from pathlib import Path

from metaheuristics.de.differential_evolution import DifferentialEvolution
from metaheuristics.problems.qap_problem import QAPProblem
from metaheuristics.metrics import CallbackMetricasDE, SurrogateDataset

class DifferentialEvolutionQAP:
    def __init__(self, **de_kwargs):
        self.de = DifferentialEvolution(**de_kwargs)

    def optimize(self, qap_path = None, mat_flujo = None, mat_dist = None, seed = 42, registrar_metricas = False, ruta_metricas = None, run_id = None):
        seed = int(seed)
        self.de.seed = seed

        # ----------------------------
        # construccion del problema
        # ----------------------------
        # se carga el problema QAP
        # o
        # se define manualmente con la matriz de flujo y distancias
        if qap_path is not None:
            problema = QAPProblem.from_qaplib(path = qap_path, seed = seed, tipo_algoritmo = "continuo")
            instancia = Path(qap_path).stem
        else:
            if mat_flujo is None or mat_dist is None:
                raise ValueError("Debes pasar qap_path o (mat_flujo, mat_dist)")
            
            problema = QAPProblem(mat_flujo = mat_flujo, mat_distancias = mat_dist, seed = seed, tipo_algoritmo = "continuo")
            instancia = "custom"
    
        # ----------------------------
        # registro de métricas (DEAP)
        # ----------------------------
        recolector = None
        callback_metricas = None

        dataset = None
        if registrar_metricas:
            from metaheuristics.metrics import RecolectorMetricasDEAP
            recolector = RecolectorMetricasDEAP()
            dataset = SurrogateDataset(
                algoritmo = "de",
                problema = "qap",
                seed = seed,
                run_info = {"instancia": str(instancia)},
            )
            tiempo_inicio = time.perf_counter()
            callback_metricas = CallbackMetricasDE(
                recolector,
                tiempo_inicio,
                lambda: self.de.evals,
                # decodifica_permutaciones = lambda pop: np.asarray(
                #     [problema.decodificar_asignacion(ind) for ind in np.asarray(pop, dtype=float)],
                #     dtype=int,
                # ),
                transforma_vectores_hamming = lambda pop: np.asarray(pop, dtype=float),
                registrar_poblacion = False,
                en_generacion = lambda g: setattr(self.de, "_generacion_actual", int(g)),
                offset_current_generation = 1,
            )

        # if registrar_metricas:
        #     recolector = RecolectorMetricasDEAP()
        #     tiempo_inicio = time.perf_counter()
        #     callback_metricas = CallbackMetricasDE(
        #         recolector,
        #         tiempo_inicio,
        #         lambda: self.de.evals,
        #         lambda pop: np.asarray(
        #             [problema.decodificar_asignacion(ind) for ind in np.asarray(pop, dtype=float)],
        #             dtype=int,
        #         ),
        #         registrar_poblacion = False,
        #     )
        

        mejor_sol, mejor_fitness = self.de.optimize(
            limites = problema.get_bounds(),
            problem = problema,
            callback_metricas = callback_metricas,
            dataset = dataset,
            perm_decodificador = lambda ind: problema.decodificar_asignacion(np.asarray(ind, dtype = float))
        )
        mejor_permutacion = problema.decodificar_asignacion(np.asarray(mejor_sol, dtype=float))

        resultado = {
            "mejor_sol": mejor_sol,
            "mejor_fitness": float(mejor_fitness),
            "mejor_permutacion": mejor_permutacion.astype(int).tolist(),
        }

        # ----------------------------
        # postprocesado de métricas
        # ----------------------------
        if registrar_metricas:
            metricas_resumen = recolector.obtener_resumen_final()
            metricas_resumen["mejor_fitness"] = float(mejor_fitness)

            metricas_logbook = recolector.obtener_logbook()
            resultado["metricas_logbook"] = metricas_logbook
            resultado["metricas_resumen"] = metricas_resumen

            # metricas
            config = getattr(callback_metricas, "config", None) or {}
            dim = int(problema.get_size())
            # tam_poblacion = int(self.de.tam_poblacion) if self.de.tam_poblacion is not None else int(10 * dim)
            evals_objetivo = config.get("max_evals")
            if evals_objetivo is None:
                evals_objetivo = int(self.de.max_evals) if self.de.max_evals is not None else int(10000 * dim)
            # f = float(self.de.f) if self.de.f is not None else 0.5
            # cr = float(self.de.cr) if self.de.cr is not None else 0.9
            # cross = str(self.de.metodo_cruce) if self.de.metodo_cruce is not None else "bin"

            evals_reales = int(self.de.evals) 
            evals_fuera_presupuesto = int(max(0, evals_reales - evals_objetivo))
            hubo_fuera_presupuesto = bool(evals_fuera_presupuesto > 0)

            if ruta_metricas is not None:
                if run_id is None:
                    run_id = f"de_qap_{instancia}_n{int(problema.get_size())}_s{seed}"
                ruta_base = Path(ruta_metricas) / run_id
                ficheros_metricas = recolector.guardar_csv_json(
                    ruta_base=ruta_base,
                    metadata={
                        "algoritmo": "de",
                        "problema": "qap",
                        "instancia": str(instancia),
                        "n": int(problema.get_size()),
                        "seed": int(seed),
                        "k_pares_hamming": int(getattr(recolector, "_k_pares_hamming", 200)),
                        "hamming_fuente": "vector_real",
                        "hamming_normalizada_01": True,
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
