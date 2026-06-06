import time
import numpy as np

class CallbackMetricasAGE:
    def __init__(self, recolector, tiempo_inicio):
        self.recolector = recolector
        self.t0 = tiempo_inicio

    def __call__(self, generacion, fitness, evaluaciones, **kwargs):
        self.recolector.registrar(
            generacion = generacion,
            fitness = fitness,
            evaluaciones = evaluaciones,
            tiempo_s=(time.perf_counter() - self.t0),
            **kwargs
        )

class CallbackMetricasDE:
    def __init__(
        self,
        recolector,
        tiempo_inicio,
        evals,
        registrar_poblacion = True,
        en_generacion = None,
        offset_current_generation = 0,
        restart_manager = None,
    ):
        self.recolector = recolector
        self.t0 = tiempo_inicio
        self.evals = evals
        self.config = None
        self.registrar_poblacion = bool(registrar_poblacion)
        self._en_generacion = en_generacion
        self.offset_current_generation = int(offset_current_generation)
        self.restart_manager = restart_manager

    def __call__(self, **estado):
        fitness = estado.get("fitness")
        if fitness is None:
            return

        # se captura la config desde el primer momento para evitar calculos posteriores
        # o en cada generacion
        if self.config is None:
            def _scalar_or_none(val):
                if val is None:
                    return None
                # SHADE pasa f y cr como arrays por individuo; tomamos la media
                arr = np.asarray(val)
                if arr.ndim > 0:
                    return float(np.mean(arr))
                return float(arr)

            self.config = {
                "population_size": int(estado.get("population_size")) if estado.get("population_size") is not None else None,
                "individual_size": int(estado.get("individual_size")) if estado.get("individual_size") is not None else None,
                "max_evals": int(estado.get("max_evals")) if estado.get("max_evals") is not None else None,
                "max_iters": int(estado.get("max_iters")) if estado.get("max_iters") is not None else None,
                "memory_size": int(estado.get("memory_size")) if estado.get("memory_size") is not None else None,
                "f": _scalar_or_none(estado.get("f")),
                "cr": _scalar_or_none(estado.get("cr")),
                "cross": str(estado.get("cross")) if estado.get("cross") is not None else None,
                "seed": int(estado.get("seed")) if estado.get("seed") is not None else None,
            }

        if estado.get("current_generation") is not None:
            gen = int(estado.get("current_generation", 0)) + self.offset_current_generation
        else:
            gen = int(estado.get("generacion", 0))
        if self._en_generacion is not None:
            self._en_generacion(gen)

        poblacion = estado.get("population")

        self.recolector.registrar(
            generacion=gen,
            fitness=np.asarray(fitness, dtype=float),
            evaluaciones=int(self.evals()),
            tiempo_s=(time.perf_counter() - self.t0),
            poblacion = (poblacion if self.registrar_poblacion else None),
        )

        if self.restart_manager is not None and estado.get("current_generation") is not None:
            aplicado = bool(self.restart_manager(estado=estado, generacion=gen))
            if aplicado:
                self.recolector.registrar(
                    generacion=gen,
                    fitness=np.asarray(estado.get("fitness"), dtype=float),
                    evaluaciones=int(self.evals()),
                    tiempo_s=(time.perf_counter() - self.t0),
                    poblacion=(estado.get("population") if self.registrar_poblacion else None),
                    sobrescribir_ultima=True,
                )


# alias retrocompatible para codigo que importaba CallbackMetricas
CallbackMetricas = CallbackMetricasAGE
