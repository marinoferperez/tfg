# implementación del algoritmo AGE combinatorio.

import numpy as np

class GeneticAlgorithmCombinatorio:
    # constructor del AGE
    # --------------------
    # construye el algoritmo con los valores utilizados comúnmente en la literatura

    def __init__(self, tam_poblacion=50, prob_cruce=0.7, prob_mutacion=0.1, tam_torneo=2, max_evals=None, seed=42):
        self.tam_poblacion = tam_poblacion
        self.prob_cruce = prob_cruce
        self.prob_mutacion = prob_mutacion
        self.tam_torneo = tam_torneo
        self.max_evals = max_evals
        self.rng = np.random.default_rng(seed)

    # mutacion_swap aplica una mutacion swap para introducir diversidad
    # intercambiando dos posiciones de la permutacion
    def mutacion_swap(self, individuo):
        mutado = individuo.copy()  # copia del individuo para no modificarlo directamente

        if self.rng.random() < self.prob_mutacion:
            i, j = self.rng.choice(len(mutado), size=2, replace=False)
            mutado[i], mutado[j] = mutado[j], mutado[i]
        return mutado

    # torneo realiza una selección por torneo de self.tam_torneo individuos DISTINTOS (replace=False)
    # escogiendo aquel de menor fitness
    def torneo(self, fitness):
        indices = self.rng.choice(len(fitness), size=self.tam_torneo, replace=False)
        return indices[np.argmin(fitness[indices])]

    # cruce_pmx recibe los dos padres seleccionados para combinar sus genes con cruce PMX
    def cruce_pmx(self, padre1, padre2):
        dim = len(padre1)
        # se eligen dos puntos de corte como fronteras en [0, dim]
        # y se usa el segmento [punto1, punto2) (fin exclusivo)
        punto1, punto2 = np.sort(self.rng.choice(dim + 1, size=2, replace=False))

        hijo1 = np.full(dim, -1, dtype=int)
        hijo2 = np.full(dim, -1, dtype=int)

        # se copia el segmento central desde cada padre
        hijo1[punto1:punto2] = padre1[punto1:punto2]
        hijo2[punto1:punto2] = padre2[punto1:punto2]

        segmento_hijo1 = set(hijo1[punto1:punto2].tolist())
        segmento_hijo2 = set(hijo2[punto1:punto2].tolist())

        mapa_hijo1 = {int(padre1[i]): int(padre2[i]) for i in range(punto1, punto2)}
        mapa_hijo2 = {int(padre2[i]): int(padre1[i]) for i in range(punto1, punto2)}

        for idx in range(dim):
            if punto1 <= idx < punto2:
                continue

            gen_hijo1 = int(padre2[idx])
            while gen_hijo1 in segmento_hijo1:
                gen_hijo1 = mapa_hijo1[gen_hijo1]
            hijo1[idx] = gen_hijo1

            gen_hijo2 = int(padre1[idx])
            while gen_hijo2 in segmento_hijo2:
                gen_hijo2 = mapa_hijo2[gen_hijo2]
            hijo2[idx] = gen_hijo2

        return hijo1, hijo2

    # optimize ejecuta AGE sobre el problema combinatorio concreto
    #   * problem : problema con método fitness
    # devuelve mejor_solucion y mejor_fitness
    def optimize(self, problem, callback_metricas = None):
        dim = int(problem.get_size())
        max_evals = self.max_evals if self.max_evals is not None else 10000 * dim

        # se crea la poblacion/solucion inicial aleatoriamente con permutaciones validas
        poblacion = np.asarray([self.rng.permutation(dim) for _ in range(self.tam_poblacion)], dtype=int)
        # se calcula el fitness de la poblacion inicial.
        fitness = np.asarray([problem.fitness(ind) for ind in poblacion], dtype=float)
        # el numero de evals va a ser igual al tamaño de la poblacion
        evals = self.tam_poblacion
        generacion = 0

        # callback (opcional) para registrar la poblacion inicial
        if callback_metricas is not None:
            callback_metricas(
                generacion = generacion,
                fitness = fitness.copy(),
                evaluaciones = evals,
            )

        # cada it evalua a 2 hijos
        while evals + 2 <= max_evals:
            # seleccion de dos padres por torneo
            idx_p1 = self.torneo(fitness)
            idx_p2 = self.torneo(fitness)
            padre1 = poblacion[idx_p1]
            padre2 = poblacion[idx_p2]

            # generacion de dos hijos con cruce_pmx restringido por prob_cruce
            # si no, se generan clones como hijos
            if self.rng.random() < self.prob_cruce:
                hijo1, hijo2 = self.cruce_pmx(padre1, padre2)
            else:
                hijo1 = padre1.copy()
                hijo2 = padre2.copy()

            # mutacion de los hijos
            hijo1 = self.mutacion_swap(hijo1)
            hijo2 = self.mutacion_swap(hijo2)

            fit_hijo1 = float(problem.fitness(hijo1))
            fit_hijo2 = float(problem.fitness(hijo2))
            evals += 2

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

            generacion += 1
            if callback_metricas is not None:
                callback_metricas(
                    generacion = generacion,
                    fitness = fitness.copy(),
                    evaluaciones = evals,
                )

        mejor_idx = int(np.argmin(fitness))
        mejor_solucion = poblacion[mejor_idx].copy()
        mejor_fitness = float(fitness[mejor_idx])

        return mejor_solucion, mejor_fitness
