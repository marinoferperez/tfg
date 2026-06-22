"""
Wrapper CEC2017 para AGE con integración online de subrogados.

Hereda de GeneticStationaryCEC2017 y sustituye el algoritmo interno por
GeneticAlgorithmContinuoOnline para filtrar candidatos mediante el subrogado
antes de consumir evaluaciones reales.
"""

import time
from pathlib import Path

import numpy as np

from src.utils.file_io import escribir_json
from src.metaheuristics.algorithms.offline.adapted.age_cec2017 import GeneticStationaryCEC2017
from src.metaheuristics.metrics.metrics_callback import CallbackMetricasAGE
from src.metaheuristics.metrics.elitist_restart import (
    construir_metadata_reinicios,
)
from src.metaheuristics.algorithms.online.age_online import GeneticAlgorithmContinuoOnline
from src.metaheuristics.surrogate.surrogate_controller import (
    ConfiguracionSubrogadoOnline,
    ControladorSubrogadoOnline,
)
from src.metaheuristics.surrogate.surrogate_stats import (
    EstadisticasSubrogado,
)
from src.benchmark.cec2017_problem import CEC2017Problem, MAX_EVALS_POR_DIM
from src.utils.experiment_io import guardar_reinicios_elitistas_csv, guardar_decisiones_subrogado_csv


class GeneticStationaryCEC2017Online(GeneticStationaryCEC2017):
    """Wrapper CEC2017 para AGE online con filtrado subrogado."""

    def __init__(self, surrogate_config=None, **age_kwargs):
        """
        Constructor del algoritmo.

        surrogate_config: ConfiguracionSubrogadoOnline compartida entre runs, o None para defaults.
        age_kwargs: argumentos para GeneticAlgorithmContinuoOnline
        """
        self.age = GeneticAlgorithmContinuoOnline(**age_kwargs)
        self.surrogate_config = surrogate_config

    def optimize(self, funcid, dim, seed=42, algname="age_online", lib_path=None, registrar_metricas=False, ruta_metricas=None, run_id=None, cec_workdir=None, guardar_decisiones_subrogado=False, guardar_reinicios_detalle=False):
        """
        Ejecuta AGE online sobre la función CEC2017 indicada.

        funcid: índice de la función CEC2017, en [1, 30].
        dim: dimensionalidad del problema.
        seed: semilla del generador aleatorio.
        algname: etiqueta para la salida de cec2017real.
        lib_path: ruta opcional a la librería compilada de CEC2017.
        registrar_metricas: si True, genera CSV/JSON de métricas.
        ruta_metricas: directorio raíz donde guardar los ficheros.
        run_id: nombre del subdirectorio de ficheros. Si es None, se genera automáticamente.
        cec_workdir: directorio de trabajo para cec2017real.
        guardar_decisiones_subrogado: si True, guarda un CSV con cada decisión del subrogado.
        guardar_reinicios_detalle: si True, guarda un CSV con el detalle de cada reinicio elitista.

        Retorna un dict con mejor_sol, mejor_fitness, mejor_error, resumen_online y, si
        registrar_metricas=True, las rutas a los ficheros generados.
        """
        seed = int(seed)
        self.age.rng = np.random.default_rng(seed)

        # construcción del problema
        problema = CEC2017Problem(funcid=funcid, dim=dim, algname=algname, lib_path=lib_path, seed=seed, workdir=cec_workdir)

        problema.enter_workdir()
        try:
            problema.prepare_run()

            # registro de métricas
            recolector = None
            callback_metricas = None

            if registrar_metricas:
                from src.metaheuristics.metrics.deap_metrics import RecolectorMetricasDEAP, guardar_metricas_deap

                recolector = RecolectorMetricasDEAP()
                tiempo_inicio = time.perf_counter()
                callback_metricas = CallbackMetricasAGE(recolector, tiempo_inicio)

            max_evals = (
                int(self.age.max_evals)
                if self.age.max_evals is not None
                else int(MAX_EVALS_POR_DIM * dim)
            )

            config_subrogado = self._configurar_subrogado(max_evals=max_evals, seed=seed)
            estadisticas_subrogado = EstadisticasSubrogado()
            # el controlador gestiona las decisiones de filtrado y las estadísticas online
            controlador_subrogado = ControladorSubrogadoOnline(config=config_subrogado, estadisticas=estadisticas_subrogado)

            # ejecución del algoritmo
            mejor_sol, mejor_fitness = self.age.optimize(limites=problema.get_bounds(), problema=problema, controlador_subrogado=controlador_subrogado, callback_metricas=callback_metricas)

            mejor_error = problema.cec_error(mejor_fitness)
            controlador_subrogado.estadisticas.registrar_resultado_final(mejor_fitness=mejor_fitness, mejor_error=mejor_error)
            resumen_online = controlador_subrogado.resumen()

            # resultado mínimo independientemente de registrar_metricas
            resultado = {
                "mejor_sol": mejor_sol,
                "mejor_fitness": float(mejor_fitness),
                "mejor_error": float(mejor_error),
                "resumen_online": resumen_online,
            }

            # postprocesado de métricas
            if registrar_metricas:
                metricas_resumen = recolector.obtener_resumen_final()
                metricas_resumen["mejor_fitness"] = float(mejor_fitness)
                metricas_resumen["mejor_error"] = float(mejor_error)

                metricas_logbook = recolector.obtener_logbook()
                resultado["metricas_logbook"] = metricas_logbook
                resultado["metricas_resumen"] = metricas_resumen

                if ruta_metricas is not None:
                    # run_id identifica esta ejecución en el arch
                    if run_id is None:
                        run_id = f"age_online_cec2017_f{int(funcid)}_d{int(dim)}_s{seed}"
                    ruta_base = Path(ruta_metricas) / run_id

                    # metadata_reinicios agrega campos de reinicio al JSON
                    metadata_reinicios = construir_metadata_reinicios(self.age.eventos_reinicio, self.age.reinicio_ratio, self.age.reinicio)

                    ficheros_metricas = guardar_metricas_deap(recolector, ruta_base=ruta_base, metadata={"algoritmo": "age_online", "problema": "cec2017", "funcid": int(funcid), "dim": int(dim), "seed": int(seed), "tam_poblacion": int(self.age.tam_poblacion), "prob_cruce": float(self.age.prob_cruce), "prob_mutacion": float(self.age.prob_mutacion), "tam_torneo": int(self.age.tam_torneo), "max_evals": int(max_evals), "sigma": float(self.age.sigma), "alpha": float(self.age.alpha), **metadata_reinicios, **resumen_online})

                    ruta_reinicios_csv = None
                    if guardar_reinicios_detalle:
                        # CSV opcional con el detalle de cada evento de reinicio elitista
                        ruta_reinicios_csv = guardar_reinicios_elitistas_csv(ruta_base, self.age.eventos_reinicio)

                    ruta_online_json = ruta_base / "resumen_online.json"
                    escribir_json(ruta_online_json, resumen_online)

                    resultado["ruta_metricas"] = str(ruta_base)
                    resultado["ficheros_metricas"] = ficheros_metricas
                    if ruta_reinicios_csv is not None:
                        resultado["ruta_reinicios_elitistas_csv"] = ruta_reinicios_csv
                    resultado["ruta_resumen_online"] = str(ruta_online_json)

            if (guardar_reinicios_detalle and ruta_metricas is not None and "ruta_reinicios_elitistas_csv" not in resultado):
                if run_id is None:
                    run_id = f"age_online_cec2017_f{int(funcid)}_d{int(dim)}_s{seed}"
                    
                ruta_base = Path(ruta_metricas) / run_id
                ruta_base.mkdir(parents=True, exist_ok=True)
                ruta_reinicios_csv = guardar_reinicios_elitistas_csv(ruta_base, self.age.eventos_reinicio)
                
                if ruta_reinicios_csv is not None:
                    resultado["ruta_reinicios_elitistas_csv"] = ruta_reinicios_csv

            if guardar_decisiones_subrogado and ruta_metricas is not None:
                if run_id is None:
                    run_id = f"age_online_cec2017_f{int(funcid)}_d{int(dim)}_s{seed}"
                ruta_base = Path(ruta_metricas) / run_id
                ruta_base.mkdir(parents=True, exist_ok=True)
                # CSV opcional con cada decisión del subrogado (aceptar/rechazar y motivo)
                ruta_decisiones_csv = guardar_decisiones_subrogado_csv(ruta_base, controlador_subrogado.estadisticas.decisiones_subrogado)
                ruta_online_json = ruta_base / "resumen_online.json"
                escribir_json(ruta_online_json, resumen_online)
                resultado["ruta_metricas"] = str(ruta_base)
                resultado["ruta_decisiones_subrogado_csv"] = ruta_decisiones_csv
                resultado["ruta_resumen_online"] = str(ruta_online_json)

            # los eventos de reinicio se devuelven siempre para facilitar el análisis
            resultado["reinicios"] = list(self.age.eventos_reinicio)
            return resultado

        finally:
            problema.exit_workdir()

    def _configurar_subrogado(self, max_evals, seed):
        """
        Reconstruye la configuración online para fijar max_evals y seed en cada run.

        max_evals: presupuesto de evaluaciones de esta ejecución concreta.
        seed: semilla del generador aleatorio.

        Retorna una ConfiguracionSubrogadoOnline lista para usar.
        """
        if self.surrogate_config is None:
            return ConfiguracionSubrogadoOnline(max_evals=int(max_evals), seed=int(seed))

        return ConfiguracionSubrogadoOnline(modelo_nombre=self.surrogate_config.modelo_nombre, modelo_params=dict(self.surrogate_config.modelo_params), cooldown_reinicio_evals=self.surrogate_config.cooldown_reinicio_evals, warmup_ratio=self.surrogate_config.warmup_ratio, window_ratio=self.surrogate_config.window_ratio, probabilidad_subrogado=self.surrogate_config.probabilidad_subrogado, max_evals=int(max_evals), minimizacion=self.surrogate_config.minimizacion, seed=int(seed), retrain_ratio=self.surrogate_config.retrain_ratio)
