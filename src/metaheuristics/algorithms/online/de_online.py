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


class DifferentialEvolutionOnline(DifferentialEvolution):
    """
    Variante online de DE/rand/1.

    Mantiene la estructura de PYADE: mutación diferencial, cruce binomial o
    exponencial y selección uno a uno entre padre y trial. La diferencia es que
    cada trial puede filtrarse con el subrogado antes de gastar una evaluación
    real. La referencia para el filtro es el fitness real del padre
    contra el que compite el trial.
    """

    def _evaluar_individuo_real(self, problema, candidato, eval_id, generacion, dataset, controlador, evaluacion_filtrada_por_subrogado):
        """
        Evalúa realmente un trial y registra la muestra en dataset y controlador.

        problema: instancia del problema con método fitness.
        candidato: vector de prueba o miembro de la población a evaluar.
        eval_id: identificador secuencial de la evaluación.
        generacion: generación actual.
        dataset: opcional para registrar evaluaciones en el dataset de entrenamiento.
        controlador: instancia de ControladorSubrogadoOnline.
        evaluacion_filtrada_por_subrogado: True si el candidato pasó antes por el subrogado.

        Retorna el fitness real o inf si ya se agotó el presupuesto.
        """
        # PYADE puede superar max_evals porque trabaja con max_iters = max_evals / tam_poblacion
        if self._max_evals_reales is not None and self.evals >= self._max_evals_reales:
            return float("inf")

        fit = float(problema.fitness(candidato))

        if dataset is not None:
            dataset.individuo_to_dataset(
                eval_id=int(eval_id),
                generacion=int(generacion),
                x=np.asarray(candidato, dtype=float),
                fitness=float(fit),
            )

        # el controlador distingue si la evaluación fue precedida por el subrogado
        if evaluacion_filtrada_por_subrogado:
            controlador.registrar_evaluacion_tras_subrogado(candidato, fit)
        else:
            controlador.registrar_evaluacion_directa(candidato, fit)

        self.evals = controlador.evals_reales
        return fit

    def _evaluar_reinicio_real(self, candidato):
        """
        Función real usada internamente por el reinicio elitista.

        candidato: vector de decisión generado tras el reinicio.

        Los candidatos generados tras reinicio se evalúan siempre realmente y no
        pasan por el filtro subrogado.
        """
        return self._evaluar_individuo_real(
            problema=self._problema,
            candidato=candidato,
            eval_id=self.evals + 1,
            generacion=getattr(self, "_generacion_actual", 0),
            dataset=self._dataset,
            controlador=self._controlador_subrogado,
            evaluacion_filtrada_por_subrogado=False,
        )

    def _aplicar_reinicio(self, estado, generacion):
        """
        Aplica el reinicio original de DE y notifica al controlador online.

        estado: diccionario PYADE con claves population, fitness, bounds y func.
        generacion: generación actual.

        El controlador invalida el modelo actual y activa el cooldown si está configurado.
        Retorna True si se aplicó el reinicio.
        """
        aplicado = super()._aplicar_reinicio(
            estado=estado,
            generacion=generacion,
        )
        if aplicado and self._controlador_subrogado is not None:
            self._controlador_subrogado.registrar_reinicio()
        return aplicado

    def optimize(self, limites, problema, controlador_subrogado, callback_metricas=None, dataset=None):
        """
        Ejecuta DE online con filtrado subrogado sobre el problema indicado.

        limites: array (dim, 2) con [inferior, superior] por variable.
        problema: instancia del problema con método fitness.
        controlador_subrogado: instancia de ControladorSubrogadoOnline.
        callback_metricas: opcional para registrar métricas por generación.
        dataset: opcional para registrar evaluaciones en el dataset de entrenamiento.

        Retorna (mejor_solucion, mejor_fitness).
        """
        if controlador_subrogado is None:
            raise ValueError("DE online requiere un controlador_subrogado.")

        limites = np.asarray(limites, dtype=float)
        dim = int(limites.shape[0])

        self.evals = 0
        self.eventos_reinicio = []
        self._problema = problema
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

        # se extraen los parámetros resueltos que usará el bucle manual
        population_size = int(params["population_size"])
        individual_size = int(params["individual_size"])
        max_evals = int(params["max_evals"])
        f = float(params["f"])
        cr = float(params["cr"])
        cross = str(params["cross"])
        seed = int(params["seed"])
        bounds = np.asarray(params["bounds"], dtype=float)

        self._max_evals_reales = max_evals
        self._gestor_reinicio = self._crear_gestor_reinicio(max_evals)

        np.random.seed(seed)
        population = pyade_commons.init_population(population_size, individual_size, bounds)

        # evaluación inicial de la población completa sin filtro subrogado
        fitness = []
        for candidato in population:
            fit = self._evaluar_individuo_real(
                problema=problema,
                candidato=candidato,
                eval_id=self.evals + 1,
                generacion=0,
                dataset=dataset,
                controlador=controlador_subrogado,
                evaluacion_filtrada_por_subrogado=False,
            )
            fitness.append(fit)
        fitness = np.asarray(fitness, dtype=float)

        max_iters = max(1, max_evals // population_size)
        # el subrogado puede rechazar trials sin consumir evaluaciones, por eso
        # el número de generaciones puede superar max_iters; este límite evita bucles infinitos
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

            # fitness de los trials inicializado a inf; se actualiza solo si se evalúan
            c_fitness = np.full(population_size, float("inf"), dtype=float)

            for idx, trial in enumerate(crossed):
                if self.evals >= max_evals:
                    break

                decision = controlador_subrogado.decidir_evaluacion(
                    candidato=trial,
                    fitness_ref=float(fitness[idx]),
                    generacion=generacion,
                )

                # el subrogado puede rechazar el trial sin consumir evaluación real
                if not decision.debe_evaluar:
                    continue

                c_fitness[idx] = self._evaluar_individuo_real(
                    problema=problema,
                    candidato=trial,
                    eval_id=self.evals + 1,
                    generacion=generacion,
                    dataset=dataset,
                    controlador=controlador_subrogado,
                    evaluacion_filtrada_por_subrogado=decision.uso_subrogado,
                )

            # selección uno a uno: el trial reemplaza al padre solo si mejora su fitness
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
