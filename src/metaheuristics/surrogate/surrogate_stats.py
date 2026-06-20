"""
Registro de estadisticas para la integracion online con subrogados.

Este modulo acumula contadores de una ejecucion online:
    - candidatos generados;
    - candidatos filtrados por el subrogado;
    - candidatos rechazados;
    - candidatos evaluados realmente;
    - tiempos de entrenamiento y prediccion;
    - reinicios;
    - calidad final.

No decide si un candidato se evalua ni entrena el modelo.
"""

from dataclasses import dataclass, field

@dataclass
class EstadisticasSubrogado:
    """
    Contadores agregados de una ejecucion online.

    dataclass porque los contadores se actualizan incrementalmente durante la ejecucion de la metaheuristica.
    """
    
    evals_reales: int = 0

    candidatos_generados: int = 0
    candidatos_con_subrogado: int = 0
    candidatos_evaluados_reales: int = 0
    candidatos_aceptados_por_subrogado: int = 0
    candidatos_rechazados: int = 0

    entrenamientos_modelo: int = 0
    tiempo_entrenamiento_total: float = 0.0
    tiempo_prediccion_total: float = 0.0

    reinicios: int = 0

    mejor_fitness: float | None = None
    mejor_error: float | None = None
    
    decisiones_subrogado: list[dict] = field(default_factory=list)

    def registrar_candidato_generado(self):
        """Incrementa el contador de candidatos producidos por la metaheurística."""
        self.candidatos_generados += 1

    def registrar_evaluacion_directa(self):
        """
        Candidato evaluado directamente sin pasar por el subrogado.
        """
        self.candidatos_evaluados_reales += 1
        self.evals_reales += 1

    def registrar_candidato_con_subrogado(self):
        """Incrementa el contador de candidatos para los que se consultó el subrogado."""
        self.candidatos_con_subrogado += 1
        
    def registrar_aceptado_por_subrogado(self):
        """
        El subrogado permite evaluar realmente el candidato.

        No significa que el candidato haya sido aceptado en la poblacion, solo significa que ha pasado el filtro
        """
        self.candidatos_aceptados_por_subrogado += 1
        
    def registrar_evaluacion_tras_subrogado(self):
        """Candidato que pasó el filtro del subrogado y fue evaluado realmente."""
        self.candidatos_evaluados_reales += 1
        self.evals_reales += 1

    def registrar_rechazo_subrogado(self):
        """Candidato rechazado por el subrogado sin consumir evaluación real."""
        self.candidatos_rechazados += 1

    def registrar_entrenamiento(self, tiempo_segundos):
        """
        Registra un entrenamiento del modelo y su duración.

        tiempo_segundos: tiempo en segundos que tardó el ajuste del modelo.
        """
        self.entrenamientos_modelo += 1
        self.tiempo_entrenamiento_total += float(tiempo_segundos)

    def registrar_prediccion(self, tiempo_segundos):
        """
        Registra el tiempo de una predicción del modelo.

        tiempo_segundos: tiempo en segundos que tardó la predicción puntual.
        """
        self.tiempo_prediccion_total += float(tiempo_segundos)

    def registrar_reinicio(self):
        """Incrementa el contador de reinicios de la metaheurística."""
        self.reinicios += 1

    def registrar_resultado_final(self, mejor_fitness, mejor_error=None):
        """
        Almacena el mejor fitness y error al final de la ejecución.

        mejor_fitness: mejor valor de la función objetivo encontrado en el run.
        mejor_error: distancia al óptimo conocido (f - f*), o None si no se dispone de él.
        """
        self.mejor_fitness = float(mejor_fitness)
        self.mejor_error = None if mejor_error is None else float(mejor_error)

    def registrar_decision_subrogado(self, **payload):
        """
        Registra una decision tomada con el subrogado.

        **payload: campos de la decisión: evals_reales, generacion, reinicios, evals_desde_reinicio,
            fitness_pred, fitness_ref, margen_pred_ref, debe_evaluar, motivo.

        Permite analizar fitness_pred - fitness_ref y detectar si el filtro está rechazando
        candidatos por poco margen.
        """
        self.decisiones_subrogado.append(dict(payload))
        
    @property
    def porcentaje_rechazo(self):
        """Fracción de candidatos filtrados por el subrogado que fueron rechazados."""
        if self.candidatos_con_subrogado == 0:
            return 0.0
        return self.candidatos_rechazados / self.candidatos_con_subrogado

    @property
    def tiempo_online_total(self):
        """Suma del tiempo de entrenamiento y predicción del subrogado."""
        return self.tiempo_entrenamiento_total + self.tiempo_prediccion_total

    def resumen(self):
        """
        Devuelve un diccionario con las estadisticas agregadas.
        """
        return {
            "evals_reales": int(self.evals_reales),
            "candidatos_generados": int(self.candidatos_generados),
            "candidatos_con_subrogado": int(self.candidatos_con_subrogado),
            "candidatos_evaluados_reales": int(self.candidatos_evaluados_reales),
            "candidatos_aceptados_por_subrogado": int(self.candidatos_aceptados_por_subrogado),
            "candidatos_rechazados": int(self.candidatos_rechazados),
            "porcentaje_rechazo": float(self.porcentaje_rechazo),
            "entrenamientos_modelo": int(self.entrenamientos_modelo),
            "tiempo_entrenamiento_total": float(self.tiempo_entrenamiento_total),
            "tiempo_prediccion_total": float(self.tiempo_prediccion_total),
            "tiempo_online_total": float(self.tiempo_online_total),
            "reinicios": int(self.reinicios),
            "mejor_fitness": self.mejor_fitness,
            "mejor_error": self.mejor_error,
        }
