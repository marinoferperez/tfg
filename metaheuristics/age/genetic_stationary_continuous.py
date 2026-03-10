# implementación del algoritmo AGE para problemas continuos.

import numpy as np
from metaheuristics.metrics.surrogate_dataset import SurrogateDataset

class GeneticAlgorithmContinuo:
    # constructor del AGE
    # --------------------
    # construye el algoritmo con los valores utilizados comúnmente en la literatura

    def __init__(self, tam_poblacion = 50, prob_cruce = 0.7, prob_mutacion = 0.1, sigma = 0.3, alpha = 0.45, tam_torneo = 2, max_evals = None, seed = 42):
        self.tam_poblacion = tam_poblacion
        self.prob_cruce = prob_cruce
        self.prob_mutacion = prob_mutacion
        self.sigma = sigma
        self.alpha = alpha
        self.tam_torneo = tam_torneo
        self.max_evals = max_evals
        self.rng = np.random.default_rng(seed)
    
    # mutacion_gaussiana aplica la mutacion gaussiana por gen 
    # para introducir diversidad (ruido)
    # se garantiza que la sol sigue dentro de los limites
    def mutacion_gaussiana(self, individuo, limites):
        mutado = individuo.copy() # copia del individuo para no modificarlo directamente

        mask = self.rng.random(len(individuo)) < self.prob_mutacion
        noise = self.rng.normal(0, self.sigma, size=len(individuo)) # se introduce ruido
        mutado[mask] += noise[mask] # solo se mutan los genes seleccionados 
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
        hijo1 = self.rng.uniform(low, high)
        hijo2 = self.rng.uniform(low, high)

        hijo1 = np.clip(hijo1, limites[:, 0], limites[:, 1])
        hijo2 = np.clip(hijo2, limites[:, 0], limites[:, 1])
        return hijo1, hijo2

    # optimize ejecuta AGE sobre el problema concreto
    #   * limites : array (dim, 2) con [min, max] por dim
    #   * problem : problema con método fitness
    # devuelve mejor_solucion y mejor_fitness

    def optimize(self, limites, problem, callback_metricas = None, dataset = None):
        limites = np.asarray(limites, dtype=float)
        dim = limites.shape[0]
        max_evals = self.max_evals if self.max_evals is not None else 10000 * dim

        # se crea la poblacion/solucion inicial aleatoriamente respetando los limites
        poblacion = problem.create_population(self.rng, self.tam_poblacion, dim, limites)
        
        # al ser la version continua, no tenemos que guardar ninguna permutacion
        fitness_list = []
        for ind in poblacion:
            fit = float(problem.fitness(ind))
            fitness_list.append(fit)
            if dataset is not None:
                dataset.individuo_to_dataset(
                    eval_id = len(fitness_list),
                    generacion = 0,
                    x = ind,
                    fitness = fit
                )

        # se calcula el fitness de la poblacion inicial.
        fitness = np.asarray(fitness_list, dtype=float)
        # el numero de evals va a ser igual al tamaño de la poblacion
        evals = self.tam_poblacion
        generacion = 0

        # callback (opcional) para registrar la poblacion inicial
        if callback_metricas is not None:
            callback_metricas(
                generacion = generacion,
                fitness = fitness.copy(),
                evaluaciones = evals,
                poblacion = np.asarray(poblacion).copy(),
            )

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

            fit_hijo1 = float(problem.fitness(hijo1))
            if dataset is not None:
                dataset.individuo_to_dataset(
                    eval_id = evals + 1,    # ids de eval unicos x muestra en cada it
                    generacion = generacion,
                    x = hijo1,
                    fitness = fit_hijo1
                )

            fit_hijo2 = float(problem.fitness(hijo2))
            if dataset is not None:
                dataset.individuo_to_dataset(
                    eval_id = evals + 2,
                    generacion = generacion,
                    x = hijo2,
                    fitness = fit_hijo2
                )

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
                    fitness = fitness.copy(),     # se pasa una copia ya que en cada iteracion el fitness cambia
                    evaluaciones = evals,
                    poblacion = np.asarray(poblacion).copy(),
                )

        mejor_idx = int(np.argmin(fitness))
        mejor_solucion = poblacion[mejor_idx].copy()
        mejor_fitness = float(fitness[mejor_idx])

        return mejor_solucion, mejor_fitness
