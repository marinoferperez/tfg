"""
Adaptación de DE al benchmark continuo CEC2017 (cec2017real).

CEC2017 es un problema de minimización, y DE se ejecuta en ese mismo criterio.
"""

import numpy as np
import time
from pathlib import Path

from metaheuristics.de.differential_evolution import DifferentialEvolution
from metaheuristics.problems.cec2017_problem import CEC2017Problem

class DifferentialEvolutionCEC2017:
    def __init__(self, **de_kwargs):
        self.de = DifferentialEvolution(**de_kwargs)

    def optimize(
        self,
        funcid,
        dim,
        seed=42,
        algname="de_cec",
        lib_path=None,
        registrar_metricas=False,
        ruta_metricas=None,
        run_id=None,
    ):
        seed = int(seed)
        self.de.seed = seed

        problema = CEC2017Problem(funcid=funcid, dim=dim, algname=algname, lib_path=lib_path, seed=seed)
        problema.prepare_run()

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

        mejor_error = problema.cec_error(mejor_fitness)

        resultado = {
            "mejor_sol": mejor_sol,
            "mejor_fitness": float(mejor_fitness),
            "mejor_error": mejor_error,
        }

        if registrar_metricas:
            metricas_logbook = recolector.obtener_logbook_serializable()
            metricas_resumen = recolector.obtener_resumen_final()
            metricas_resumen["mejor_fitness"] = float(mejor_fitness)
            metricas_resumen["mejor_error"] = float(mejor_error)
            resultado["metricas_logbook"] = metricas_logbook
            resultado["metricas_resumen"] = metricas_resumen

            if ruta_metricas is not None:
                if run_id is None:
                    run_id = f"de_cec2017_f{int(funcid)}_d{int(dim)}_s{seed}"
                ruta_base = Path(ruta_metricas) / run_id
                ficheros_metricas = recolector.guardar_csv_json(
                    ruta_base=ruta_base,
                    metadata={
                        "algoritmo": "de",
                        "problema": "cec2017",
                        "funcid": int(funcid),
                        "dim": int(dim),
                        "seed": int(seed),
                        "algname": str(algname),
                    },
                )
                resultado["ruta_metricas"] = str(ruta_base)
                resultado["ficheros_metricas"] = ficheros_metricas

        return resultado
