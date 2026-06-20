"""
Variante online de SHADE con filtrado subrogado.

Extiende SHADE con soporte para filtrar trial vectors mediante un modelo
subrogado antes de consumir evaluaciones reales de la función objetivo.
La adaptación histórica de F y CR y la selección uno a uno se mantienen
idénticas a la implementación PYADE original.
"""

import random

import numpy as np
import scipy.stats

from src.metaheuristics.algorithms.offline.shade import SHADE, pyade_commons, pyade_shade

class SHADEOnline(SHADE):
    """
    Variante online de SHADE.

    Mantiene la estructura de PYADE: adaptación histórica de F y CR, mutación
    current-to-pbest, cruce binomial y selección uno a uno entre padre y trial.
    La diferencia es que cada trial puede filtrarse con el subrogado antes de
    consumir una evaluación real. La referencia para el filtro es el
    fitness real del padre contra el que compite el trial.
    """

    def _evaluar_individuo_real(self, problema, candidato, eval_id, generacion, dataset, controlador, evaluacion_filtrada_por_subrogado):
        """
        Evalúa realmente un trial mediante la función objetivo y registra la muestra.

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
            dataset.registrar_evaluacion(
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

        Los candidatos generados tras reinicio siempre se evalúan realmente y no
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
        Aplica el reinicio original de SHADE y notifica al controlador online.

        estado: diccionario PYADE con claves population, fitness, bounds, func y archive.
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
        Ejecuta SHADE online con filtrado subrogado sobre el problema indicado.

        limites: array (dim, 2) con [inferior, superior] por variable.
        problema: instancia del problema con método fitness.
        controlador_subrogado: instancia de ControladorSubrogadoOnline.
        callback_metricas: opcional para registrar métricas por generación.
        dataset: opcional para registrar evaluaciones en el dataset de entrenamiento.

        Retorna (mejor_solucion, mejor_fitness).
        """
        if controlador_subrogado is None:
            raise ValueError("SHADE online requiere un controlador_subrogado.")

        limites = np.asarray(limites, dtype=float)
        dim = int(limites.shape[0])

        self.evals = 0
        self.eventos_reinicio = []
        self._problema = problema
        self._dataset = dataset
        self._controlador_subrogado = controlador_subrogado
        self._generacion_actual = 0

        params = pyade_shade.get_default_params(dim=dim)
        params["individual_size"] = dim
        params["bounds"] = limites
        params["seed"] = int(self.seed)

        if self.tam_poblacion is not None:
            params["population_size"] = int(self.tam_poblacion)
        if self.max_evals is not None:
            params["max_evals"] = int(self.max_evals)
        if self.memory_size is not None:
            params["memory_size"] = int(self.memory_size)

        if int(params["max_evals"]) <= 0:
            raise ValueError("max_evals debe ser > 0")

        # se extraen los parámetros resueltos que usará el bucle manual
        population_size = int(params["population_size"])
        individual_size = int(params["individual_size"])
        max_evals = int(params["max_evals"])
        memory_size = int(params["memory_size"])
        seed = int(params["seed"])
        bounds = np.asarray(params["bounds"], dtype=float)

        self._max_evals_reales = max_evals
        self._gestor_reinicio = self._crear_gestor_reinicio(max_evals)

        # PYADE usa ambos generadores en SHADE
        np.random.seed(seed)
        random.seed(seed)

        # inicialización original de SHADE: población, memorias de CR y F, archivo
        population = pyade_commons.init_population(population_size, individual_size, bounds)
        m_cr = np.ones(memory_size) * 0.5
        m_f = np.ones(memory_size) * 0.5
        archive = []
        k = 0

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
                memory_size=memory_size,
                seed=seed,
                f=None,
                cr=None,
            )

        generacion = 0
        all_indexes = list(range(memory_size))

        while self.evals < max_evals and generacion < max_generaciones_online:
            generacion += 1
            self._generacion_actual = generacion

            # adaptación de CR y F siguiendo la implementación PYADE de SHADE
            r = np.random.choice(all_indexes, population_size)

            cr = np.random.normal(m_cr[r], 0.1, population_size)
            cr = np.clip(cr, 0, 1)
            cr[cr == 1] = 0

            f = scipy.stats.cauchy.rvs(loc=m_f[r], scale=0.1, size=population_size)
            f[f > 1] = 0

            while np.sum(f <= 0) != 0:
                r_invalidos = np.random.choice(all_indexes, int(np.sum(f <= 0)))
                f[f <= 0] = scipy.stats.cauchy.rvs(
                    loc=m_f[r_invalidos],
                    scale=0.1,
                    size=int(np.sum(f <= 0)),
                )

            p = np.random.uniform(low=2 / population_size, high=0.2, size=population_size)

            mutated = pyade_commons.current_to_pbest_mutation(
                population,
                fitness,
                f.reshape(len(f), 1),
                p,
                bounds,
            )
            crossed = pyade_commons.crossover(
                population,
                mutated,
                cr.reshape(len(f), 1),
            )

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

            # adaptación histórica de SHADE: se actualiza m_cr y m_f con los éxitos de esta generación
            archive.extend(population[indexes])

            if len(indexes) > 0:
                if len(archive) > memory_size:
                    archive = random.sample(archive, memory_size)

                if np.max(cr) != 0:
                    weights = np.abs(fitness[indexes] - c_fitness[indexes])
                    weights /= np.sum(weights)
                    m_cr[k] = np.sum(weights * cr[indexes])
                else:
                    m_cr[k] = 1

                m_f[k] = np.sum(f[indexes] ** 2) / np.sum(f[indexes])

                k += 1
                if k == memory_size:
                    k = 0

            fitness[indexes] = c_fitness[indexes]
            estado = {
                "population_size": population_size,
                "individual_size": individual_size,
                "bounds": bounds,
                "func": self._evaluar_reinicio_real,
                "opts": params.get("opts"),
                "memory_size": memory_size,
                "callback": callback_metricas,
                "max_evals": max_evals,
                "seed": seed,
                "population": population,
                "m_cr": m_cr,
                "m_f": m_f,
                "archive": archive,
                "k": k,
                "fitness": fitness,
                "all_indexes": all_indexes,
                "max_iters": max_iters,
                "current_generation": generacion - 1,
                "r": r,
                "cr": cr,
                "f": f,
                "p": p,
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
