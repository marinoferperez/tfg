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
        decodifica_permutaciones = None,
        registrar_poblacion = True,
        en_generacion = None,
        transforma_vectores_hamming = None,
    ):
        self.recolector = recolector
        self.t0 = tiempo_inicio
        self.evals = evals
        self.config = None
        self.decodifica_permutaciones = decodifica_permutaciones
        self.registrar_poblacion = bool(registrar_poblacion)
        self.transforma_vectores_hamming = transforma_vectores_hamming
        self._en_generacion = en_generacion

    def __call__(self, **estado):
        fitness = estado.get("fitness")
        if fitness is None:
            return

        # se captura la config desde el primer momento para evitar calculos posteriores
        # o en cada generacion
        if self.config is None:
            self.config = {
                "population_size": int(estado.get("population_size")) if estado.get("population_size") is not None else None,
                "individual_size": int(estado.get("individual_size")) if estado.get("individual_size") is not None else None,
                "max_evals": int(estado.get("max_evals")) if estado.get("max_evals") is not None else None,
                "max_iters": int(estado.get("max_iters")) if estado.get("max_iters") is not None else None,
                "f": float(estado.get("f")) if estado.get("f") is not None else None,
                "cr": float(estado.get("cr")) if estado.get("cr") is not None else None,
                "cross": str(estado.get("cross")) if estado.get("cross") is not None else None,
                "seed": int(estado.get("seed")) if estado.get("seed") is not None else None,
            }

        # gen = int(estado.get("current_generation", 0))
        gen = int(estado.get("current_generation", estado.get("generacion", 0)))
        if self._en_generacion is not None:
            self._en_generacion(gen)

        poblacion = estado.get("population")

        permutaciones = None
        if self.decodifica_permutaciones is not None and poblacion is not None:
            permutaciones = self.decodifica_permutaciones(poblacion)

        vectores_hamming = None
        if self.transforma_vectores_hamming is not None and poblacion is not None:
            vectores_hamming = self.transforma_vectores_hamming(poblacion)

        self.recolector.registrar(
            generacion=gen,
            fitness=np.asarray(fitness, dtype=float),
            evaluaciones=int(self.evals()),
            tiempo_s=(time.perf_counter() - self.t0),
            poblacion = (poblacion if self.registrar_poblacion else None),
            permutaciones = permutaciones,
            vectores_hamming = vectores_hamming,
        )


# alias retrocompatible para codigo que importaba CallbackMetricas
CallbackMetricas = CallbackMetricasAGE
