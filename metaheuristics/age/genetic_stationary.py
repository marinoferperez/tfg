
import numpy as np


# ---------------------------------------------------------------------------
#  Clase GeneticAlgorithm
# ---------------------------------------------------------------------------

class GeneticAlgorithm:
    """
    Algoritmo Genético Estacionario (AGE).
    
    Encapsula todos los parámetros, operadores y la lógica del algoritmo genético.
    """
    
    def __init__(self, tam_poblacion=50, prob_cruce=0.7, prob_mutacion=0.1,
                 sigma=0.3, alpha=0.45, tam_torneo=2, max_evals=None, seed=42):
        """
        Constructor del AGE.
        
        :param tam_poblacion: Tamaño de la población.
        :param prob_cruce: Probabilidad de cruce.
        :param mutation_rate: Prob. mutación por gen (None = 1/dim, se calcula después).
        :param sigma: Desviación típica mutación gaussiana.
        :param alpha: Parámetro α para BLX-α (valor fijo entre 0.3-0.5).
        :param tournament_size: Tamaño del torneo.
        :param max_evals: Evaluaciones máximas (None = 10000*dim, se calcula después).
        :param seed: Semilla aleatoria.
        """
        self.tam_poblacion = tam_poblacion
        self.prob_cruce = prob_cruce
        self.prob_mutacion = prob_mutacion
        self.sigma = sigma
        self.alpha = alpha
        self.tam_torneo = tam_torneo
        self.max_evals = max_evals
        self.seed = seed
        
        # RNG propio para reproducibilidad
        self.rng = np.random.default_rng(seed)
    
    def mutacion_gaussiana(self, instancia, prob_mutacion, limites):
        """
        Mutación gaussiana gen a gen.

        Para cada gen, con probabilidad *prob_mutacion*, se le suma un valor
        N(0, σ).  El resultado se recorta a los límites.

        :return: Individuo (posiblemente) mutado.
        """
        mutado = instancia.copy()
        mask = self.rng.random(len(instancia)) < prob_mutacion
        noise = self.rng.normal(0, self.sigma, size=len(instancia))
        mutado[mask] += noise[mask]
        mutado = np.clip(mutado, limites[:, 0], limites[:, 1])
        return mutado

    def torneo(self, fitness):
        """
        Selección por torneo de tamaño *tam_torneo*.

        Criterio fijo: minimización (CEC2017).

        :return: Índice del individuo ganador.
        """
        indices = self.rng.choice(len(fitness), size=self.tam_torneo, replace=False)
        mejor_idx = indices[np.argmin(fitness[indices])]
        return mejor_idx

    def cruce_blx(self, padre1, padre2, limites):
        """
        Cruce BLX-α.  Para cada gen *i*:
            d = |p1_i - p2_i|
            hijo_i ~ U[min(p1_i, p2_i) - α·d,  max(p1_i, p2_i) + α·d]

        Los hijos se recortan a los límites del problema.

        :return: Tupla (hijo1, hijo2).
        """
        d = np.abs(padre1 - padre2)
        low = np.minimum(padre1, padre2) - self.alpha * d
        high = np.maximum(padre1, padre2) + self.alpha * d

        hijo1 = self.rng.uniform(low, high)
        hijo2 = self.rng.uniform(low, high)

        # Clip a los límites del problema
        hijo1 = np.clip(hijo1, limites[:, 0], limites[:, 1])
        hijo2 = np.clip(hijo2, limites[:, 0], limites[:, 1])

        return hijo1, hijo2
    
    def optimize(self, limites, problem):
        """
        Ejecuta el AGE sobre el problema dado.
        
        :param limites: Array (dim, 2) con [min, max] por gen.
        :param problem: Instancia de problema con método fitness().
        :return: Tupla (mejor_solución, mejor_fitness).
        """
        dim = limites.shape[0]
        max_evals = self.max_evals if self.max_evals is not None else 10000 * dim
        
        #### POBLACION INICIAL ####   
        # generamos la poblacion
        poblacion = problem.create_population(self.rng, self.tam_poblacion, dim, limites)
        # se calcula el fitness de la poblacion generada
        fitness_eval = problem.fitness(poblacion)
        fitness = np.asarray(fitness_eval, dtype=float)
        
        if fitness.shape == ():
            fitness = np.asarray([problem.fitness(ind) for ind in poblacion], dtype=float)
        else:
            fitness = fitness.reshape(-1)
        if fitness.shape[0] != self.tam_poblacion:
            raise ValueError("problem.fitness(population) debe devolver un fitness por individuo")
        # el numero de evals realizadas hasta ahora va a ser igual al tamaño de la poblacion ya que
        # para cada individuo generado (son 50) se ha evaluado su fitness.
        evals = self.tam_poblacion
        

        while evals + 2 <= max_evals:
            # seleccion de dos padres por torneo 
            idx_p1 = self.torneo(fitness)
            idx_p2 = self.torneo(fitness)
            
            padre1 = poblacion[idx_p1]
            padre2 = poblacion[idx_p2]
            
            # cruzamos los hijos con probabilidad "cross_rate"
            if self.rng.random() < self.prob_cruce:
                hijo1, hijo2 = self.cruce_blx(padre1, padre2, limites)
            else:
                hijo1 = padre1.copy()
                hijo2 = padre2.copy()
            
            # mutacion de los hijos con probabilidad "mutation_rate"
            hijo1 = self.mutacion_gaussiana(hijo1, self.prob_mutacion, limites)
            hijo2 = self.mutacion_gaussiana(hijo2, self.prob_mutacion, limites)
            
            # evaluamos los hijos
            fit_hijo1 = float(problem.fitness(hijo1))
            fit_hijo2 = float(problem.fitness(hijo2))
            evals += 2  # cada vez que evaluamos se incrementa el contador
            
            # seleccionamos mejor hijo de los dos en cuanto a su fitness
            if fit_hijo1 < fit_hijo2:
                mejor_hijo, mejor_fitness = hijo1, fit_hijo1
            else:
                mejor_hijo, mejor_fitness = hijo2, fit_hijo2
            
            # reemplazo estacionario - seleccionamos el peor de la poblacion y lo sustituimos por el mejor hijo
            peor_idx = np.argmax(fitness)

            if mejor_fitness < fitness[peor_idx]:
                poblacion[peor_idx] = mejor_hijo
                fitness[peor_idx] = mejor_fitness
        
        # buscamos el mejor individuo de la poblacion
        mejor_idx = np.argmin(fitness)
        return poblacion[mejor_idx].copy(), fitness[mejor_idx]
