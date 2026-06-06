# implementación del algoritmo AGE para problemas continuos (CEC2017 únicamente).
#
# Diferencias respecto a la versión anterior (general):
#   - La población inicial se genera directamente con rng.uniform sobre los
#     límites del dominio, sin delegar en problem.create_population.
#   - El dataset de surrogate ya no serializa 'generacion'; esa información se
#     utiliza solo durante la ejecución para resumir luego la diversidad por
#     generación en el HDF5 final.

import numpy as np

from metaheuristics.metrics.reinicio_elitista import (
    ControlReinicioElitista,
    calcular_diversidad_normalizada,
    seleccionar_indice_elitista,
)


class GeneticAlgorithmContinuo:
    # constructor del AGE
    # --------------------
    # construye el algoritmo con los valores utilizados comúnmente en la literaturaxs

    def __init__(self, tam_poblacion=50, prob_cruce=0.7, prob_mutacion=0.1,
                 sigma=0.3, alpha=0.45, tam_torneo=2, max_evals=None, seed=42,
                 reinicio_elitista=False,
                 reinicio_elitista_ratio_estabilidad_diversidad=None,
                 reinicio_elitista_ratio_paciencia=0.05,
                 reinicio_elitista_ventana_evaluaciones=2500):
        self.tam_poblacion = tam_poblacion
        self.prob_cruce = prob_cruce
        self.prob_mutacion = prob_mutacion
        self.sigma = sigma
        self.alpha = alpha
        self.tam_torneo = tam_torneo
        self.max_evals = max_evals
        self.rng = np.random.default_rng(seed)
        self.reinicio_elitista = bool(
            reinicio_elitista or reinicio_elitista_ratio_estabilidad_diversidad is not None
        )
        self.reinicio_elitista_ratio_estabilidad_diversidad = (
            float(reinicio_elitista_ratio_estabilidad_diversidad)
            if reinicio_elitista_ratio_estabilidad_diversidad is not None
            else None
        )
        self.reinicio_elitista_ratio_paciencia = float(reinicio_elitista_ratio_paciencia)
        self.reinicio_elitista_ventana_evaluaciones = int(reinicio_elitista_ventana_evaluaciones)
        self.eventos_reinicio_elitista = []
        self._control_reinicio_elitista = (
            ControlReinicioElitista(
                self.reinicio_elitista_ratio_estabilidad_diversidad,
                max_evals=max_evals,
                ratio_paciencia=self.reinicio_elitista_ratio_paciencia,
                ventana_evaluaciones=self.reinicio_elitista_ventana_evaluaciones,
            )
            if self.reinicio_elitista
            else None
        )

    def _generar_poblacion_uniforme(self, limites, n_individuos):
        return self.rng.uniform(limites[:, 0], limites[:, 1], size=(int(n_individuos), limites.shape[0]))

    def _evaluar_individuo(self, problem, individuo, eval_id, generacion, dataset=None):
        fit = float(problem.fitness(individuo))
        if dataset is not None:
            dataset.individuo_to_dataset(
                eval_id=int(eval_id),
                generacion=int(generacion),
                x=individuo,
                fitness=fit,
            )
        return fit

    def _diversidad_normalizada(self, poblacion, rango_inf=-100.0, rango_sup=100.0):
        return calcular_diversidad_normalizada(poblacion)

    def _aplicar_reinicio_elitista(self, poblacion, fitness, limites, problem, evals, generacion, max_evals, dataset=None):
        if self._control_reinicio_elitista is None:
            return poblacion, fitness, evals, False

        if not self._control_reinicio_elitista.debe_reiniciar(
            fitness=fitness,
            poblacion=poblacion,
            evaluaciones=int(evals),
            generacion=generacion,
            bounds=limites,
        ):
            return poblacion, fitness, evals, False
        diagnostico = self._control_reinicio_elitista.diagnostico_reinicio()

        n_nuevos = int(self.tam_poblacion) - 1
        evals_restantes = int(max_evals) - int(evals)
        if n_nuevos <= 0 or evals_restantes < n_nuevos:
            return poblacion, fitness, evals, False

        elite_idx = seleccionar_indice_elitista(fitness)
        elite = np.asarray(poblacion[elite_idx], dtype=float).copy()
        elite_fit = float(fitness[elite_idx])
        mejor_fit = float(elite_fit)

        nueva_poblacion = np.empty_like(poblacion)
        nuevo_fitness = np.empty_like(fitness)
        nueva_poblacion[0] = elite
        nuevo_fitness[0] = elite_fit

        nuevos = self._generar_poblacion_uniforme(limites, n_nuevos)
        for idx, individuo in enumerate(nuevos, start=1):
            nuevo_eval_id = int(evals) + idx
            nueva_poblacion[idx] = individuo
            nuevo_fitness[idx] = self._evaluar_individuo(
                problem,
                individuo,
                eval_id=nuevo_eval_id,
                generacion=generacion,
                dataset=dataset,
            )

        evals_despues = int(evals) + n_nuevos
        self._control_reinicio_elitista.registrar_estado_post_reinicio(
            fitness=nuevo_fitness,
            evaluaciones=int(evals_despues),
        )
        evento = dict(diagnostico)
        evento.update(
            {
                "generacion": int(generacion),
                "evaluaciones_antes_reinicio": int(evals),
                "evaluaciones_despues_reinicio": int(evals_despues),
                "indice_individuo_preservado": int(elite_idx),
                "fitness_preservado": float(elite_fit),
                "mejor_fitness": float(mejor_fit),
            }
        )
        self.eventos_reinicio_elitista.append(evento)
        return nueva_poblacion, nuevo_fitness, evals_despues, True

    # mutacion_gaussiana aplica la mutacion gaussiana por gen
    # para introducir diversidad (ruido)
    # se garantiza que la sol sigue dentro de los limites
    def mutacion_gaussiana(self, individuo, limites):
        mutado = individuo.copy()  # copia del individuo para no modificarlo directamente

        mask = self.rng.random(len(individuo)) < self.prob_mutacion
        noise = self.rng.normal(0, self.sigma, size=len(individuo))  # se introduce ruido
        mutado[mask] += noise[mask]  # solo se mutan los genes seleccionados
        return np.clip(mutado, limites[:, 0], limites[:, 1])

    # torneo realiza una selección por torneo de self.tam_torneo individuos DISTINTOS (replace=False)
    # escogiendo aquel de menor fitness
    def torneo(self, fitness):
        indices = self.rng.choice(len(fitness), size=self.tam_torneo, replace=False)
        return indices[np.argmin(fitness[indices])]

    # cruce_blx recibe los dos padres seleccionados para combinar sus genes con cruce blx
    # recorta en los limites
    def cruce_blx(self, padre1, padre2, limites):
        # calcula la distancia por gen entre padres
        d = np.abs(padre1 - padre2)

        # se define el intervalo (low, high)
        low = np.minimum(padre1, padre2) - self.alpha * d
        high = np.maximum(padre1, padre2) + self.alpha * d

        # se muestrean los hijos dentro del nuevo intervalo
        hijo1 = np.clip(self.rng.uniform(low, high), limites[:, 0], limites[:, 1])
        hijo2 = np.clip(self.rng.uniform(low, high), limites[:, 0], limites[:, 1])
        return hijo1, hijo2

    # optimize ejecuta AGE sobre el problema concreto
    #   * limites : array (dim, 2) con [min, max] por dim
    #   * problem : problema con método fitness
    # devuelve mejor_solucion y mejor_fitness

    def optimize(self, limites, problem, callback_metricas=None, dataset=None):
        limites = np.asarray(limites, dtype=float)
        dim = limites.shape[0]
        max_evals = self.max_evals if self.max_evals is not None else 10000 * dim
        self.eventos_reinicio_elitista = []
        if self.reinicio_elitista:
            self._control_reinicio_elitista = ControlReinicioElitista(
                self.reinicio_elitista_ratio_estabilidad_diversidad,
                max_evals=max_evals,
                ratio_paciencia=self.reinicio_elitista_ratio_paciencia,
                ventana_evaluaciones=self.reinicio_elitista_ventana_evaluaciones,
            )
        else:
            self._control_reinicio_elitista = None

        # población inicial uniforme sobre el dominio continuo
        poblacion = self._generar_poblacion_uniforme(limites, self.tam_poblacion)

        fitness_list = []
        for ind in poblacion:
            fit = self._evaluar_individuo(
                problem,
                ind,
                eval_id=len(fitness_list) + 1,
                generacion=0,
                dataset=dataset,
            )
            fitness_list.append(fit)

        # se calcula el fitness de la poblacion inicial.
        fitness = np.asarray(fitness_list, dtype=float)
        # el numero de evals va a ser igual al tamaño de la poblacion
        evals = self.tam_poblacion

        # callback (opcional) para registrar la poblacion inicial
        if callback_metricas is not None:
            callback_metricas(
                generacion=0,
                fitness=fitness.copy(),
                evaluaciones=evals,
                poblacion=poblacion.copy(),
            )

        generacion = 0
        # cada it evalua a 2 hijos
        while evals + 2 <= max_evals:
            # seleccion de dos padres por torneo
            idx_p1 = self.torneo(fitness)
            idx_p2 = self.torneo(fitness)
            padre1 = poblacion[idx_p1]
            padre2 = poblacion[idx_p2]

            # generacion de dos hijos con cruce_blx restringido por prob_cruce
            # si no, se generan clones como hijos
            if self.rng.random() < self.prob_cruce:
                hijo1, hijo2 = self.cruce_blx(padre1, padre2, limites)
            else:
                hijo1 = padre1.copy()
                hijo2 = padre2.copy()

            # mutacion de los hijos
            hijo1 = self.mutacion_gaussiana(hijo1, limites)
            hijo2 = self.mutacion_gaussiana(hijo2, limites)

            fit_hijo1 = self._evaluar_individuo(
                problem,
                hijo1,
                eval_id=evals + 1,
                generacion=generacion + 1,
                dataset=dataset,
            )

            fit_hijo2 = self._evaluar_individuo(
                problem,
                hijo2,
                eval_id=evals + 2,
                generacion=generacion + 1,
                dataset=dataset,
            )

            evals += 2
            generacion += 1

            # se elige el mejor hijo (minimizacion)
            if fit_hijo1 < fit_hijo2:
                mejor_hijo, mejor_fit = hijo1, fit_hijo1
            else:
                mejor_hijo, mejor_fit = hijo2, fit_hijo2

            # reemplazo estacionario:
            # si el fitness del peor individuo de la pob actual es peor que el fitness del mejor hijo,
            # se reemplaza el peor individuo por el mejor hijo
            peor_idx = int(np.argmax(fitness))
            if mejor_fit < fitness[peor_idx]:
                poblacion[peor_idx] = mejor_hijo
                fitness[peor_idx] = mejor_fit

            if callback_metricas is not None:
                callback_metricas(
                    generacion=generacion,
                    fitness=fitness.copy(),   # se pasa una copia ya que en cada iteracion el fitness cambia
                    evaluaciones=evals,
                    poblacion=poblacion.copy(),
                )

            poblacion, fitness, evals, reiniciado = self._aplicar_reinicio_elitista(
                poblacion,
                fitness,
                limites,
                problem,
                evals,
                generacion,
                max_evals,
                dataset=dataset,
            )
            if reiniciado and callback_metricas is not None:
                callback_metricas(
                    generacion=generacion,
                    fitness=fitness.copy(),
                    evaluaciones=evals,
                    poblacion=poblacion.copy(),
                    sobrescribir_ultima=True,
                )

        mejor_idx = int(np.argmin(fitness))
        return poblacion[mejor_idx].copy(), float(fitness[mejor_idx])
