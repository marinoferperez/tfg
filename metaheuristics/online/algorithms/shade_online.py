import random

import numpy as np
import scipy.stats

from metaheuristics.metrics.reinicio_elitista import ControlReinicioElitista
from metaheuristics.offline.algorithms.shade import SHADE, pyade_commons, pyade_shade

class SHADEOnline(SHADE):
    """
    Variante online de SHADE.

    Mantiene la estructura de PYADE:
        - adaptacion historica de F y CR;
        - mutacion current-to-pbest;
        - cruce binomial;
        - seleccion uno a uno entre padre y trial.

    La diferencia es que cada trial puede filtrarse con el subrogado antes de
    consumir una evaluacion real. Igual que en DE, la referencia natural es el
    fitness real del padre contra el que compite el trial.
    """
    
    def _evaluar_individuo_real(self, problem, individuo, eval_id, generacion, dataset, controlador, evaluacion_filtrada_por_subrogado,
    ):
        """
        Evalua realmente un individuo mediante la función objetivo real y registra la muestra.

        Si el candidato paso antes por el subrogado, se registra como evaluacion
        tras filtro; si no, se registra como evaluacion directa.
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
        Funcion real usada internamente por el reinicio elitista.

        Los individuos generados tras reinicio siempre se evaluan realmente y no
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
        Aplica el reinicio original de SHADE y notifica al controlador online.

        Notificar el reinicio es importante porque invalida el modelo actual y
        activa el cooldown si esta configurado.
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
        Ejecuta SHADE con filtro subrogado online.

        En cada generacion se generan vectores de prueba (trials). Para cada trial:
            1. Se compara la prediccion del subrogado contra el fitness real del padre.
            2. Si el subrogado lo rechaza, no se evalua realmente.
            3. Si pasa el filtro, se evalua con CEC2017.
            4. La seleccion final siempre usa fitness real.
        """
        if controlador_subrogado is None:
            raise ValueError("SHADE online requiere un controlador_subrogado.")

        limites = np.asarray(limites, dtype=float)
        dim = int(limites.shape[0])

        self.evals = 0
        self.eventos_reinicio = []
        self._problema = problem
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

        population_size = int(params["population_size"])
        individual_size = int(params["individual_size"])
        max_evals = int(params["max_evals"])
        memory_size = int(params["memory_size"])
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

        # PYADE usa ambos generadores en SHADE.
        np.random.seed(seed)
        random.seed(seed)

        # Inicializacion original de SHADE.
        population = pyade_commons.init_population(population_size, individual_size, bounds)
        m_cr = np.ones(memory_size) * 0.5
        m_f = np.ones(memory_size) * 0.5
        archive = []
        k = 0

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

        # Como el subrogado puede rechazar trials sin consumir evaluaciones, el
        # numero de generaciones puede ser mayor que max_iters. Este limite evita
        # bucles muy largos si p es alta y el filtro rechaza demasiado.
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

            # Adaptacion de CR y F siguiendo PYADE.
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

            # Adaptacion historica de SHADE siguiendo la implementacion original de PYADE.
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
