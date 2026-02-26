"""
Adaptacion de DE al problema combinatorio QAP.

QAP se modela en minimizacion.
"""

import numpy as np
import time
from pathlib import Path

from metaheuristics.de.differential_evolution import DifferentialEvolution
from metaheuristics.problems.qap_problem import QAPProblem

class DifferentialEvolutionQAP:
    def __init__(self, **de_kwargs):
        self.de = DifferentialEvolution(**de_kwargs)

    def optimize(
        self,
        qap_path=None,
        flow_matrix=None,
        distance_matrix=None,
        seed=42,
        registrar_metricas=False,
        ruta_metricas=None,
        run_id=None,
    ):
        seed = int(seed)
        self.de.seed = seed

        if qap_path is not None:
            problema = QAPProblem.from_qaplib(path=qap_path, seed=seed, tipo_algoritmo="continuo")
            instancia = Path(qap_path).stem
        else:
            if flow_matrix is None or distance_matrix is None:
                raise ValueError("Debes pasar qap_path o (flow_matrix, distance_matrix)")
            problema = QAPProblem(mat_flujo=flow_matrix, mat_distancias=distance_matrix, seed=seed, tipo_algoritmo="continuo")
            instancia = "custom"

        recolector = None
        callback_metricas = None
        tiempo_inicio = time.perf_counter()

        if registrar_metricas:
            try:
                from metaheuristics.metrics import RecolectorMetricasDEAP
            except ModuleNotFoundError as exc:
                if str(exc).find("deap") != -1:
                    raise ModuleNotFoundError(
                        "No se pudo importar 'deap'. Instala dependencias con: "
                        "python3 -m pip install -r requirements.txt"
                    ) from exc
                raise

            recolector = RecolectorMetricasDEAP()

            def callback_metricas(generacion, fitness, evaluaciones):
                recolector.registrar(
                    generacion=generacion,
                    fitness_vector=fitness,
                    evaluaciones=evaluaciones,
                    tiempo_s=(time.perf_counter() - tiempo_inicio),
                    mejor_hasta_ahora=float(np.min(fitness)),
                )

        mejor_sol, mejor_fitness = self.de.optimize(
            limites=problema.get_bounds(),
            problem=problema,
            callback_metricas=callback_metricas,
        )
        mejor_permutacion = problema.decodificar_asignacion(np.asarray(mejor_sol, dtype=float))

        resultado = {
            "mejor_sol": mejor_sol,
            "mejor_fitness": float(mejor_fitness),
            "best_qap_cost": float(mejor_fitness),
            "mejor_permutacion": mejor_permutacion.astype(int).tolist(),
        }

        if registrar_metricas:
            metricas_logbook = recolector.obtener_logbook_serializable()
            metricas_resumen = recolector.obtener_resumen_final()
            metricas_resumen["mejor_fitness"] = float(mejor_fitness)
            metricas_resumen["best_qap_cost"] = float(mejor_fitness)
            resultado["metricas_logbook"] = metricas_logbook
            resultado["metricas_resumen"] = metricas_resumen

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
                    },
                )
                resultado["ruta_metricas"] = str(ruta_base)
                resultado["ficheros_metricas"] = ficheros_metricas

        return resultado
