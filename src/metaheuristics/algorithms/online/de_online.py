"""
Variante online de DE/rand/1 con filtrado subrogado.

Extiende DifferentialEvolution con soporte para filtrar trial vectors mediante
un modelo subrogado antes de consumir evaluaciones reales de la función objetivo.
"""

import numpy as np

from src.metaheuristics.algorithms.offline.de import (
    DifferentialEvolution,
    pyade_commons,
    pyade_de,
)
from src.metaheuristics.metrics.elitist_restart import ControlReinicioElitista

# aceptar trial si f_predicha(trial) < f_real(padre)
# rechazar si f_predicha(trial) >= f_real(padre)

# se respeta la logica original de DE, donde cada trial compite contra su vector padre

class DifferentialEvolutionOnline(DifferentialEvolution):
    """
    Variante online de DE/rand/1.

    Mantiene la estructura de PYADE:
        - mutacion diferencial;
        - cruce binomial o exponencial;
        - seleccion uno a uno entre padre y trial.

    La diferencia es que cada trial puede filtrarse con el subrogado antes de
    gastar una evaluacion real. En DE la referencia natural es el fitness real
    del padre contra el que compite el trial.
    """

    def _evaluar_individuo_real(self, problem, individuo, eval_id, generacion, dataset, controlador,
        evaluacion_filtrada_por_subrogado,
    ):
        """
        Evalúa realmente un trial y registra la muestra en dataset y controlador.

        Si el candidato pasó antes por el subrogado, se registra como evaluación
        tras filtro; si no, se registra como evaluación directa.
        Retorna inf si ya se agotó el presupuesto.
        """
        if self._max_evals_reales is not None and self.evals >= self._max_evals_reales:
            return float("inf")

        fit = float(problem.fitness(individuo))

        if dataset is not None:
            dataset.individuo_to_dataset(
                eval_id=int(eval_id),
                generacion=int(generacion),
                x=np.asarray(individuo, dtype=float),
                fitness=float(fit),
            )

        if evaluacion_filtrada_por_subrogado:
            controlador.registrar_evaluacion_tras_subrogado(individuo, fit)
        else:
            controlador.registrar_evaluacion_directa(individuo, fit)

        self.evals = controlador.evals_reales
        return fit

    def _evaluar_reinicio_real(self, individuo):
        """
        Función real usada internamente por el reinicio elitista.

        Los individuos generados tras reinicio se evalúan siempre realmente y no
        pasan por el filtro subrogado.
        """
        return self._evaluar_individuo_real(
            problem=self._problema,
            individuo=individuo,
            eval_id=self.evals + 1,
            generacion=getattr(self, "_generacion_actual", 0),
            dataset=self._dataset,
            controlador=self._controlador_subrogado,
            evaluacion_filtrada_por_subrogado=False,
        )

    def _aplicar_reinicio(self, estado, generacion):
        """
        Aplica el reinicio original de DE y notifica al controlador online.

        Notificar el reinicio es importante porque invalida el modelo actual y
        activa el cooldown si está configurado.
        """
        aplicado = super()._aplicar_reinicio(
            estado=estado,
            generacion=generacion,
        )
        if aplicado and self._controlador_subrogado is not None:
            self._controlador_subrogado.registrar_reinicio()
        return aplicado

    def optimize(self, limites, problem, controlador_subrogado, callback_metricas=None, dataset=None):
        """
        Ejecuta DE online con filtrado subrogado sobre el problema indicado.

        limites: array (dim, 2) con los límites del dominio.
        problem: problema con método fitness.
        controlador_subrogado: instancia de ControladorSubrogadoOnline.
        callback_metricas: función opcional invocada al final de cada generación.
        dataset: SurrogateDataset opcional para registrar evaluaciones reales.

        Retorna (mejor_solucion, mejor_fitness).
        """
        if controlador_subrogado is None:
            raise ValueError("DE online requiere un controlador_subrogado.")

        limites = np.asarray(limites, dtype=float)
        dim = int(limites.shape[0])

        self.evals = 0
        self.eventos_reinicio = []
        self._problema = problem
        self._dataset = dataset
        self._controlador_subrogado = controlador_subrogado
        self._generacion_actual = 0

        params = pyade_de.get_default_params(dim=dim)
        params["individual_size"] = dim
        params["bounds"] = limites
        params["seed"] = int(self.seed)

        if self.tam_poblacion is not None:
            params["population_size"] = int(self.tam_poblacion)
        if self.max_evals is not None:
            params["max_evals"] = int(self.max_evals)
        if self.f is not None:
            params["f"] = float(self.f)
        if self.cr is not None:
            params["cr"] = float(self.cr)
        if self.metodo_cruce is not None:
            params["cross"] = self.metodo_cruce

        if int(params["max_evals"]) <= 0:
            raise ValueError("max_evals debe ser > 0")

        population_size = int(params["population_size"])
        individual_size = int(params["individual_size"])
        max_evals = int(params["max_evals"])
        f = float(params["f"])
        cr = float(params["cr"])
        cross = str(params["cross"])
        seed = int(params["seed"])
        bounds = np.asarray(params["bounds"], dtype=float)

        self._max_evals_reales = max_evals
        if self.reinicio:
            self._gestor_reinicio = ControlReinicioElitista(
                max_evals=max_evals,
                ratio_paciencia=self.reinicio_ratio,
            )
        else:
            self._gestor_reinicio = None

        np.random.seed(seed)
        population = pyade_commons.init_population(population_size, individual_size, bounds)

        fitness = []
        for individuo in population:
            fit = self._evaluar_individuo_real(
                problem=problem,
                individuo=individuo,
                eval_id=self.evals + 1,
                generacion=0,
                dataset=dataset,
                controlador=controlador_subrogado,
                evaluacion_filtrada_por_subrogado=False,
            )
            fitness.append(fit)
        fitness = np.asarray(fitness, dtype=float)

        max_iters = max(1, max_evals // population_size)
        max_generaciones_online = max_iters * 10

        if callback_metricas is not None:
            callback_metricas(
                generacion=0,
                population=population,
                fitness=fitness,
                population_size=population_size,
                individual_size=individual_size,
                max_evals=max_evals,
                max_iters=max_iters,
                f=f,
                cr=cr,
                cross=cross,
                seed=seed,
            )

        generacion = 0
        while self.evals < max_evals and generacion < max_generaciones_online:
            generacion += 1
            self._generacion_actual = generacion

            mutated = pyade_commons.binary_mutation(population, f, bounds)
            if cross == "bin":
                crossed = pyade_commons.crossover(population, mutated, cr)
            else:
                crossed = pyade_commons.exponential_crossover(population, mutated, cr)

            c_fitness = np.full(population_size, float("inf"), dtype=float)

            for idx, trial in enumerate(crossed):
                if self.evals >= max_evals:
                    break

                decision = controlador_subrogado.decidir_evaluacion(
                    candidato=trial,
                    fitness_ref=float(fitness[idx]),
                    generacion=generacion,
                )

                if not decision.debe_evaluar:
                    continue

                c_fitness[idx] = self._evaluar_individuo_real(
                    problem=problem,
                    individuo=trial,
                    eval_id=self.evals + 1,
                    generacion=generacion,
                    dataset=dataset,
                    controlador=controlador_subrogado,
                    evaluacion_filtrada_por_subrogado=decision.uso_subrogado,
                )

            population, indexes = pyade_commons.selection(
                population,
                crossed,
                fitness,
                c_fitness,
                return_indexes=True,
            )
            fitness[indexes] = c_fitness[indexes]

            estado = {
                "population_size": population_size,
                "individual_size": individual_size,
                "f": f,
                "cr": cr,
                "bounds": bounds,
                "func": self._evaluar_reinicio_real,
                "opts": params.get("opts"),
                "callback": callback_metricas,
                "cross": cross,
                "max_evals": max_evals,
                "seed": seed,
                "population": population,
                "fitness": fitness,
                "max_iters": max_iters,
                "current_generation": generacion - 1,
                "mutated": mutated,
                "crossed": crossed,
                "c_fitness": c_fitness,
                "indexes": indexes,
            }

            if callback_metricas is not None:
                callback_metricas(**estado)
            elif self.reinicio:
                self._aplicar_reinicio(
                    estado=estado,
                    generacion=generacion,
                )

        best = int(np.argmin(fitness))
        return population[best].copy(), float(fitness[best])
