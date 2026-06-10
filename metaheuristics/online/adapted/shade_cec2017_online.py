import json
import time
from pathlib import Path

from metaheuristics.metrics import CallbackMetricasDE, SurrogateDataset
from metaheuristics.metrics.reinicio_elitista import (
    construir_metadata_reinicios,
    guardar_reinicios_elitistas_csv,
)
from metaheuristics.online.algorithms.shade_online import SHADEOnline
from metaheuristics.online.surrogate_controller import (
    ConfiguracionSubrogadoOnline,
    ControladorSubrogadoOnline,
)
from metaheuristics.online.surrogate_stats import (
    EstadisticasSubrogado,
    guardar_decisiones_subrogado_csv,
)
from metaheuristics.cec2017 import CEC2017Problem
from metaheuristics.offline.adapted.shade_cec2017 import SHADECEC2017


class SHADECEC2017Online(SHADECEC2017):
    """
    Wrapper CEC2017 para SHADE online.

    Hereda del wrapper CEC2017 original, pero sustituye el algoritmo interno por
    SHADEOnline para poder filtrar trial vectors con el subrogado.
    """

    def __init__(self, surrogate_config=None, **shade_kwargs):
        self.shade = SHADEOnline(**shade_kwargs)
        self.surrogate_config = surrogate_config

    def optimize(
        self,
        funcid,
        dim,
        seed=42,
        lib_path=None,
        algname="shade_online",
        registrar_metricas=False,
        ruta_metricas=None,
        run_id=None,
        cec_workdir=None,
        guardar_decisiones_subrogado=False,
        guardar_dataset=True,
    ):
        seed = int(seed)
        self.shade.seed = seed

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

            recolector = None
            callback_metricas = None
            dataset = None

            if registrar_metricas:
                from metaheuristics.metrics import RecolectorMetricasDEAP

                recolector = RecolectorMetricasDEAP(filtrar_evals_no_crecientes=True)
                tiempo_inicio = time.perf_counter()
                if guardar_dataset:
                    dataset = SurrogateDataset(
                        algoritmo="shade_online",
                        problema="cec2017",
                        seed=seed,
                        run_info={"funcid": int(funcid), "dim": int(dim)},
                    )
                callback_metricas = CallbackMetricasDE(
                    recolector,
                    tiempo_inicio,
                    lambda: self.shade.evals,
                    en_generacion=lambda g: setattr(
                        self.shade,
                        "_generacion_actual",
                        int(g) + 1,
                    ),
                    offset_current_generation=1,
                    restart_manager=self.shade._aplicar_reinicio,
                )

            max_evals = (
                int(self.shade.max_evals)
                if self.shade.max_evals is not None
                else int(10000 * dim)
            )

            config_subrogado = self._configurar_subrogado(
                max_evals=max_evals,
                seed=seed,
            )
            estadisticas_subrogado = EstadisticasSubrogado()
            controlador_subrogado = ControladorSubrogadoOnline(
                config=config_subrogado,
                estadisticas=estadisticas_subrogado,
            )

            mejor_sol, mejor_fitness = self.shade.optimize(
                limites=problema.get_bounds(),
                problem=problema,
                controlador_subrogado=controlador_subrogado,
                callback_metricas=callback_metricas,
                dataset=dataset,
            )

            mejor_error = problema.cec_error(mejor_fitness)
            controlador_subrogado.estadisticas.registrar_resultado_final(
                mejor_fitness=mejor_fitness,
                mejor_error=mejor_error,
            )
            resumen_online = controlador_subrogado.resumen()

            resultado = {
                "mejor_sol": mejor_sol,
                "mejor_fitness": float(mejor_fitness),
                "mejor_error": float(mejor_error),
                "resumen_online": resumen_online,
            }

            ruta_base = None
            if ruta_metricas is not None and (
                registrar_metricas or guardar_decisiones_subrogado
            ):
                if run_id is None:
                    run_id = f"shade_online_cec2017_f{int(funcid)}_d{int(dim)}_s{seed}"
                ruta_base = Path(ruta_metricas) / run_id
                ruta_base.mkdir(parents=True, exist_ok=True)

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
                    evals_objetivo = max_evals

                evals_reales = int(self.shade.evals)
                evals_fuera_presupuesto = int(
                    max(0, evals_reales - int(evals_objetivo))
                )
                hubo_fuera_presupuesto = bool(evals_fuera_presupuesto > 0)

                if ruta_base is not None:
                    if dataset is not None:
                        rangos_generacion = dataset.obtener_rangos_generacion()
                        recolector.anotar_rangos_generacion(rangos_generacion)

                        diversidad_por_generacion = (
                            recolector.obtener_diversidad_por_generacion()
                        )
                        if 0 not in diversidad_por_generacion:
                            rango_gen0 = rangos_generacion.get(0)
                            if rango_gen0 is not None:
                                diversidad_gen0 = dataset.calcular_diversidad_rango(
                                    rango_gen0["eval_id_inicio"],
                                    rango_gen0["eval_id_fin"],
                                )
                                if diversidad_gen0 is not None:
                                    recolector.anotar_diversidad_generacion(
                                        0,
                                        diversidad_gen0,
                                    )

                        dataset.anotar_diversidad_por_generacion(
                            recolector.obtener_diversidad_por_generacion()
                        )

                    metadata_reinicios = construir_metadata_reinicios(
                        self.shade.eventos_reinicio,
                        self.shade.reinicio_ratio,
                        self.shade.reinicio,
                    )

                    ficheros_metricas = recolector.guardar_csv_json(
                        ruta_base=ruta_base,
                        metadata={
                            "algoritmo": "shade_online",
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
                            **resumen_online,
                        },
                    )

                    ruta_reinicios_csv = guardar_reinicios_elitistas_csv(
                        ruta_base,
                        self.shade.eventos_reinicio,
                    )
                    ficheros_dataset = (
                        dataset.guardar_csv_json(ruta_base)
                        if dataset is not None
                        else None
                    )

                    resultado["ficheros_metricas"] = ficheros_metricas
                    resultado["ficheros_dataset"] = ficheros_dataset
                    resultado["ruta_reinicios_elitistas_csv"] = ruta_reinicios_csv

            if ruta_base is not None:
                ruta_online_json = ruta_base / "resumen_online.json"
                ruta_online_json.write_text(
                    json.dumps(resumen_online, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                resultado["ruta_metricas"] = str(ruta_base)
                resultado["ruta_resumen_online"] = str(ruta_online_json)

                if guardar_decisiones_subrogado:
                    ruta_decisiones_csv = guardar_decisiones_subrogado_csv(
                        ruta_base,
                        controlador_subrogado.estadisticas.decisiones_subrogado,
                    )
                    resultado["ruta_decisiones_subrogado_csv"] = ruta_decisiones_csv

            resultado["reinicios"] = list(self.shade.eventos_reinicio)
            return resultado

        finally:
            problema.exit_workdir()

    def _configurar_subrogado(self, max_evals, seed):
        """
        Reconstruye la configuracion online para fijar max_evals y seed en cada run.
        """
        if self.surrogate_config is None:
            return ConfiguracionSubrogadoOnline(
                max_evals=int(max_evals),
                seed=int(seed),
            )

        return ConfiguracionSubrogadoOnline(
            modelo_nombre=self.surrogate_config.modelo_nombre,
            modelo_params=dict(self.surrogate_config.modelo_params),
            cooldown_reinicio_evals=self.surrogate_config.cooldown_reinicio_evals,
            warmup_ratio=self.surrogate_config.warmup_ratio,
            window_ratio=self.surrogate_config.window_ratio,
            probabilidad_subrogado=self.surrogate_config.probabilidad_subrogado,
            max_evals=int(max_evals),
            minimizacion=self.surrogate_config.minimizacion,
            seed=int(seed),
            retrain_ratio=self.surrogate_config.retrain_ratio,
        )
