"""
Variante online de AGE (algoritmo genético estacionario).

Extiende GeneticoEstacionario con soporte para filtrado de candidatos mediante
un modelo subrogado. La interfaz con el subrogado se delega al ControladorSubrogadoOnline
para mantener separada la lógica de la metaheurística de la del modelo.
"""

import numpy as np
from src.benchmark.cec2017_problem import MAX_EVALS_POR_DIM

from src.metaheuristics.algorithms.offline.age import GeneticoEstacionario
from src.metaheuristics.metrics.elitist_restart import (
    seleccionar_indice_elitista,
)

class GeneticAlgorithmContinuoOnline(GeneticoEstacionario):
    """
    Variante online de AGE.

    Mantiene la lógica original del AGE estacionario: genera dos hijos, evalúa
    los que corresponda y reemplaza al peor de la población si el mejor hijo
    lo supera. La única diferencia es que antes de gastar una evaluación real
    en cada hijo se consulta al controlador subrogado, usando el peor fitness
    de la población como referencia para el filtro.
    """

    def _evaluar_individuo_real(self, problema, individuo, eval_id, generacion, dataset, controlador, evaluacion_filtrada_por_subrogado):
        """
        Evalúa realmente un individuo y registra la muestra.

        problema: instancia del problema con método fitness.
        individuo: vector de decisión.
        eval_id: identificador secuencial de la evaluación.
        generacion: generación actual.
        dataset: opcional para registrar evaluaciones en el dataset de entrenamiento.
        controlador: instancia de ControladorSubrogadoOnline.
        evaluacion_filtrada_por_subrogado: True si el individuo pasó antes por el subrogado.

        Retorna el fitness real del individuo.
        """
        fit = float(problema.fitness(individuo))

        if dataset is not None:
            dataset.registrar_evaluacion(
                eval_id=int(eval_id),
                generacion=int(generacion),
                x=np.asarray(individuo, dtype=float),
                fitness=float(fit),
            )

        # el controlador distingue si la evaluación fue precedida por el subrogado
        if evaluacion_filtrada_por_subrogado:
            controlador.registrar_evaluacion_tras_subrogado(individuo, fit)
        else:
            controlador.registrar_evaluacion_directa(individuo, fit)

        return fit

    def optimize(self, limites, problema, controlador_subrogado, callback_metricas=None, dataset=None):
        """
        Ejecuta AGE online sobre el problema indicado.

        limites: array (dim, 2) con [inferior, superior] por variable.
        problema: instancia del problema con método fitness.
        controlador_subrogado: instancia de ControladorSubrogadoOnline.
        callback_metricas: opcional para registrar métricas por generación.
        dataset: opcional para registrar evaluaciones en el dataset de entrenamiento.

        Retorna (mejor_solucion, mejor_fitness).
        """
        if controlador_subrogado is None:
            raise ValueError("AGE online requiere un controlador_subrogado.")

        limites = np.asarray(limites, dtype=float)
        dim = limites.shape[0]
        max_evals = int(self.max_evals if self.max_evals is not None else MAX_EVALS_POR_DIM * dim)

        self.eventos_reinicio = []
        self._gestor_reinicio = self._crear_gestor_reinicio(max_evals)

        poblacion = self._generar_poblacion_uniforme(limites, self.tam_poblacion)

        # evaluación inicial de la población completa sin filtro subrogado
        fitness_list = []
        for ind in poblacion:
            eval_id = controlador_subrogado.evals_reales + 1
            fit = self._evaluar_individuo_real(
                problema=problema,
                individuo=ind,
                eval_id=eval_id,
                generacion=0,
                dataset=dataset,
                controlador=controlador_subrogado,
                evaluacion_filtrada_por_subrogado=False,
            )
            fitness_list.append(fit)

        fitness = np.asarray(fitness_list, dtype=float)
        evals = controlador_subrogado.evals_reales

        if callback_metricas is not None:
            callback_metricas(
                generacion=0,
                fitness=fitness.copy(),
                evaluaciones=evals,
                poblacion=poblacion.copy(),
            )

        generacion = 0

        while evals < max_evals:
            idx_p1 = self.torneo(fitness)
            idx_p2 = self.torneo(fitness)
            padre1 = poblacion[idx_p1]
            padre2 = poblacion[idx_p2]

            if self.rng.random() < self.prob_cruce:
                hijo1, hijo2 = self.cruce_blx(padre1, padre2, limites)
            else:
                hijo1 = padre1.copy()
                hijo2 = padre2.copy()

            hijo1 = self.mutacion_gaussiana(hijo1, limites)
            hijo2 = self.mutacion_gaussiana(hijo2, limites)

            generacion += 1

            # el peor individuo actual sirve de referencia para el filtro subrogado
            peor_idx_inicial = int(np.argmax(fitness))
            fitness_ref = float(fitness[peor_idx_inicial])
            hijos_evaluados = []

            for hijo in (hijo1, hijo2):
                if evals >= max_evals:
                    break

                decision = controlador_subrogado.decidir_evaluacion(
                    candidato=hijo,
                    fitness_ref=fitness_ref,
                    generacion=generacion,
                )

                # el subrogado puede rechazar el candidato sin consumir evaluación real
                if not decision.debe_evaluar:
                    continue

                eval_id = evals + 1
                fit_hijo = self._evaluar_individuo_real(
                    problema=problema,
                    individuo=hijo,
                    eval_id=eval_id,
                    generacion=generacion,
                    dataset=dataset,
                    controlador=controlador_subrogado,
                    evaluacion_filtrada_por_subrogado=decision.uso_subrogado,
                )

                evals = controlador_subrogado.evals_reales
                hijos_evaluados.append((hijo, fit_hijo))

            if not hijos_evaluados:
                continue

            # de los hijos evaluados se escoge el de menor fitness
            mejor_hijo, mejor_fit = min(hijos_evaluados, key=lambda item: item[1])

            # reemplazo estacionario: el mejor hijo desplaza al peor si lo supera
            peor_idx = int(np.argmax(fitness))
            if mejor_fit < fitness[peor_idx]:
                poblacion[peor_idx] = mejor_hijo
                fitness[peor_idx] = mejor_fit

            if callback_metricas is not None:
                callback_metricas(
                    generacion=generacion,
                    fitness=fitness.copy(),
                    evaluaciones=evals,
                    poblacion=poblacion.copy(),
                )

            poblacion, fitness, evals, reiniciado = self._aplicar_reinicio_online(
                poblacion=poblacion,
                fitness=fitness,
                limites=limites,
                problema=problema,
                evals=evals,
                generacion=generacion,
                max_evals=max_evals,
                dataset=dataset,
                controlador=controlador_subrogado,
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

    def _aplicar_reinicio_online(self, poblacion, fitness, limites, problema, evals, generacion, max_evals, dataset, controlador):
        """
        Aplica reinicio elitista para AGE online.

        poblacion: array con la población actual.
        fitness: array de fitness de la población actual.
        limites: array (dim, 2) con [inferior, superior] por variable.
        problema: instancia del problema con método fitness.
        evals: evaluaciones reales consumidas hasta el momento.
        generacion: generación actual.
        max_evals: presupuesto máximo de evaluaciones.
        dataset: opcional para registrar evaluaciones en el dataset de entrenamiento.
        controlador: instancia de ControladorSubrogadoOnline.

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

        # se necesita al menos un individuo nuevo y presupuesto suficiente
        n_nuevos = int(self.tam_poblacion) - 1
        evals_restantes = int(max_evals) - int(evals)
        if n_nuevos <= 0 or evals_restantes < n_nuevos:
            return poblacion, fitness, evals, False

        # se preserva el mejor individuo (élite) en la primera posición
        elite_idx = seleccionar_indice_elitista(fitness)
        elite = np.asarray(poblacion[elite_idx], dtype=float).copy()
        elite_fit = float(fitness[elite_idx])

        nueva_poblacion = np.empty_like(poblacion)
        nuevo_fitness = np.empty_like(fitness)
        nueva_poblacion[0] = elite
        nuevo_fitness[0] = elite_fit

        # el resto de la población se regenera uniformemente y se evalúa con la función real
        nuevos = self._generar_poblacion_uniforme(limites, n_nuevos)

        for idx, individuo in enumerate(nuevos, start=1):
            eval_id = controlador.evals_reales + 1
            nueva_poblacion[idx] = individuo
            nuevo_fitness[idx] = self._evaluar_individuo_real(
                problema=problema,
                individuo=individuo,
                eval_id=eval_id,
                generacion=generacion,
                dataset=dataset,
                controlador=controlador,
                evaluacion_filtrada_por_subrogado=False,
            )

        evals_despues = controlador.evals_reales

        self._gestor_reinicio.registrar_estado_post_reinicio(
            fitness=nuevo_fitness,
            evaluaciones=int(evals_despues),
        )

        # se registra el evento de reinicio para trazabilidad
        evento = dict(diagnostico)
        evento.update(
            {
                "generacion": int(generacion),
                "evals_antes_reinicio": int(evals),
                "evals_despues_reinicio": int(evals_despues),
                "indice_individuo_preservado": int(elite_idx),
                "fitness_preservado": float(elite_fit),
                "mejor_fitness": float(elite_fit),
            }
        )
        self.eventos_reinicio.append(evento)
        # el controlador invalida el modelo actual y activa el cooldown si corresponde
        controlador.registrar_reinicio()

        return nueva_poblacion, nuevo_fitness, evals_despues, True
