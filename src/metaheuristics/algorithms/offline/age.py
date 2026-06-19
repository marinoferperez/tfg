"""
Algoritmo genético estacionario (AGE) para optimización continua.

Implementa un genético estacionario con operadores BLX-α y mutación gaussiana.
Opcionalmente activa el mecanismo de reinicio elitista por estancamiento del
segundo mejor individuo mediante ControlReinicioElitista.
"""

import numpy as np
from src.benchmark.cec2017_problem import MAX_EVALS_POR_DIM

from src.metaheuristics.metrics.elitist_restart import (
    ControlReinicioElitista,
    seleccionar_indice_elitista,
)

class GeneticoEstacionario:
    """
    Algoritmo genético estacionario para optimización continua.

    En cada generación produce dos hijos, evalúa ambos y reemplaza al peor
    individuo de la población si el mejor hijo mejora su fitness.
    """

    def __init__(self, tam_poblacion=50, prob_cruce=0.7, prob_mutacion=0.1,
                 sigma=0.3, alpha=0.45, tam_torneo=2, max_evals=None,
                 reinicio=False, reinicio_ratio=0.05):
        """
        Construye el algoritmo con los valores utilizados comúnmente en la literatura.

        tam_poblacion: número de individuos de la población.
        prob_cruce: probabilidad de aplicar cruce BLX entre dos padres.
        prob_mutacion: probabilidad de mutar cada gen.
        sigma: desviación típica de la mutación gaussiana.
        alpha: factor de expansión del intervalo en el cruce BLX.
        tam_torneo: tamaño del torneo para selección de padres.
        max_evals: presupuesto máximo de evaluaciones. Si es None, se usa MAX_EVALS_POR_DIM * dim.
        reinicio: activa el mecanismo de reinicio elitista.
        reinicio_ratio: fracción de max_evals usada como ventana de paciencia.
        """
        self.tam_poblacion = tam_poblacion
        self.prob_cruce = prob_cruce
        self.prob_mutacion = prob_mutacion
        self.sigma = sigma
        self.alpha = alpha
        self.tam_torneo = tam_torneo
        self.max_evals = max_evals
        self.rng = np.random.default_rng()
        self.reinicio = bool(reinicio)
        self.reinicio_ratio = float(reinicio_ratio)
        self.eventos_reinicio = []
        self._gestor_reinicio = (
            ControlReinicioElitista(
                max_evals=max_evals,
                ratio_paciencia=self.reinicio_ratio,
            )
            if self.reinicio
            else None
        )

    def _generar_poblacion_uniforme(self, limites, n_individuos):
        """
        Genera individuos muestreados de forma uniforme dentro del dominio.

        limites: array (dim, 2) con [inferior, superior] por variable.
        n_individuos: número de soluciones a generar.
        """
        return self.rng.uniform(limites[:, 0], limites[:, 1], size=(int(n_individuos), limites.shape[0]))

    def _evaluar_individuo(self, problem, individuo, eval_id, generacion, dataset=None):
        """
        Evalúa un individuo y, opcionalmente, lo registra en el dataset.

        problem: instancia del problema con método fitness.
        individuo: vector de decisión a evaluar.
        eval_id: identificador secuencial de la evaluación.
        generacion: generación actual del algoritmo.
        dataset: objeto opcional para guardar muestras de entrenamiento.
        """
        fit = float(problem.fitness(individuo))
        if dataset is not None:
            dataset.individuo_to_dataset(
                eval_id=int(eval_id),
                generacion=int(generacion),
                x=individuo,
                fitness=fit,
            )
        return fit

    def _aplicar_reinicio(self, poblacion, fitness, limites, problem, evals, generacion, max_evals, dataset=None):
        """
        Aplica un reinicio elitista si el controlador lo solicita.

        poblacion: matriz de individuos actuales.
        fitness: vector de fitness asociado a la población.
        limites: límites del dominio de búsqueda.
        problem: problema evaluado.
        evals: número de evaluaciones consumidas hasta el momento.
        generacion: generación actual.
        max_evals: presupuesto máximo de evaluaciones.
        dataset: dataset opcional para registrar las nuevas evaluaciones.

        Retorna (poblacion, fitness, evals, reiniciado).
        """
        if self._gestor_reinicio is None:
            return poblacion, fitness, evals, False

        if not self._gestor_reinicio.debe_reiniciar(
            fitness=fitness,
            evaluaciones=int(evals),
            generacion=generacion,
        ):
            return poblacion, fitness, evals, False
        diagnostico = self._gestor_reinicio.diagnostico_reinicio()

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
        self._gestor_reinicio.registrar_estado_post_reinicio(
            fitness=nuevo_fitness,
            evaluaciones=int(evals_despues),
        )
        evento = dict(diagnostico)
        evento.update(
            {
                "generacion": int(generacion),
                "evals_antes_reinicio": int(evals),
                "evals_despues_reinicio": int(evals_despues),
                "indice_individuo_preservado": int(elite_idx),
                "fitness_preservado": float(elite_fit),
                "mejor_fitness": float(mejor_fit),
            }
        )
        self.eventos_reinicio.append(evento)
        return nueva_poblacion, nuevo_fitness, evals_despues, True

    def mutacion_gaussiana(self, individuo, limites):
        """
        Aplica mutación gaussiana por gen para introducir diversidad.

        individuo: solución a mutar.
        limites: array (dim, 2) con los límites del dominio. La solución resultante
        se recorta para permanecer dentro de esos límites.
        """
        mutado = individuo.copy()  # copia del individuo para no modificarlo directamente

        mask = self.rng.random(len(individuo)) < self.prob_mutacion
        noise = self.rng.normal(0, self.sigma, size=len(individuo))  # se introduce ruido
        mutado[mask] += noise[mask]  # solo se mutan los genes seleccionados
        return np.clip(mutado, limites[:, 0], limites[:, 1])

    def torneo(self, fitness):
        """
        Realiza una selección por torneo y devuelve el índice del ganador.

        fitness: vector de fitness de la población actual. Se eligen
        self.tam_torneo individuos distintos y gana el de menor fitness.
        """
        indices = self.rng.choice(len(fitness), size=self.tam_torneo, replace=False)
        return indices[np.argmin(fitness[indices])]

    def cruce_blx(self, padre1, padre2, limites):
        """
        Combina dos padres mediante cruce BLX-α y recorta en los límites.

        padre1: primer progenitor.
        padre2: segundo progenitor.
        limites: array (dim, 2) con [inferior, superior] por variable.
        """
        # calcula la distancia por gen entre padres
        d = np.abs(padre1 - padre2)

        # se define el intervalo (low, high)
        low = np.minimum(padre1, padre2) - self.alpha * d
        high = np.maximum(padre1, padre2) + self.alpha * d

        # se muestrean los hijos dentro del nuevo intervalo
        hijo1 = np.clip(self.rng.uniform(low, high), limites[:, 0], limites[:, 1])
        hijo2 = np.clip(self.rng.uniform(low, high), limites[:, 0], limites[:, 1])
        return hijo1, hijo2

    def optimize(self, limites, problem, callback_metricas=None, dataset=None):
        """
        Ejecuta el genético estacionario sobre el problema concreto.

        limites: array (dim, 2) con [min, max] por dimensión.
        problem: problema con método fitness.
        callback_metricas: función opcional para registrar métricas por generación.
        dataset: dataset opcional para almacenar evaluaciones reales.

        Retorna (mejor_solucion, mejor_fitness).
        """
        limites = np.asarray(limites, dtype=float)
        dim = limites.shape[0]
        max_evals = self.max_evals if self.max_evals is not None else MAX_EVALS_POR_DIM * dim
        self.eventos_reinicio = []
        if self.reinicio:
            self._gestor_reinicio = ControlReinicioElitista(
                max_evals=max_evals,
                ratio_paciencia=self.reinicio_ratio,
            )
        else:
            self._gestor_reinicio = None

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

            poblacion, fitness, evals, reiniciado = self._aplicar_reinicio(
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
