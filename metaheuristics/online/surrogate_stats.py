# Debe guardar los contadores online. Como registrar cada candidato puede generar ficheros muy grandes, conviene separar dos niveles:

# - resumen por ejecución;
# - trazas detalladas solo para pruebas pequeñas o depuración.

# Métricas agregadas por run:

# ```text
# evals_reales
# candidatos_generados
# candidatos_con_subrogado
# candidatos_evaluados_directamente
# candidatos_rechazados
# candidatos_aceptados_por_subrogado
# porcentaje_rechazo
# entrenamientos_rbf
# tiempo_entrenamiento_total
# tiempo_prediccion_total
# reinicios
# mejor_fitness
# mejor_error
# ```


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

from pathlib import Path
from dataclasses import dataclass, field

from preprocesado_de_datos.utils.path_utils import escribir_csv_dicts

@dataclass
class EstadisticasSubrogado:
    """
    Contadores agregados de una ejecucion online.

    Se mantiene como dataclass mutable porque los contadores se actualizan
    incrementalmente durante la ejecucion de la metaheuristica.
    """
    
    evals_reales: int = 0

    candidatos_generados: int = 0
    candidatos_con_subrogado: int = 0
    candidatos_evaluados_reales: int = 0
    candidatos_aceptados_por_subrogado: int = 0
    candidatos_rechazados: int = 0

    entrenamientos_rbf: int = 0
    tiempo_entrenamiento_total: float = 0.0
    tiempo_prediccion_total: float = 0.0

    reinicios: int = 0

    mejor_fitness: float | None = None
    mejor_error: float | None = None
    
    # Trazas ligeras. No se guardan todos los candidatos por defecto para evitar # ficheros enormes en experimentos completos.
    eventos: list[dict] = field(default_factory=list)
    decisiones_subrogado: list[dict] = field(default_factory=list)

    def registrar_candidato_generado(self):
        self.candidatos_generados += 1
    
    def registrar_evaluacion_real(self):
        self.evals_reales += 1
        
    def registrar_evaluacion_directa(self):
        """
        Candidato evaluado directamente sin pasar por el subrogado.

        Esto ocurre durante el calentamiento inicial o cuando la probabilidad p
        decide no aplicar el filtro.
        """
        self.candidatos_evaluados_reales += 1
        self.evals_reales += 1

    def registrar_candidato_con_subrogado(self):
        self.candidatos_con_subrogado += 1
        
    def registrar_aceptado_por_subrogado(self):
        """
        El subrogado permite evaluar realmente el candidato.

        No significa que el candidato haya sido aceptado en la poblacion.
        Solo significa que ha pasado el filtro
        """
        self.candidatos_aceptados_por_subrogado += 1
        
    def registrar_evaluacion_tras_subrogado(self):
        self.candidatos_evaluados_reales += 1
        self.evals_reales += 1

    def registrar_rechazo_subrogado(self):
        self.candidatos_rechazados += 1
        
    def registrar_entrenamiento(self, tiempo_segundos: float):
        self.entrenamientos_rbf += 1
        self.tiempo_entrenamiento_total += float(tiempo_segundos)

    def registrar_prediccion(self, tiempo_segundos: float):
        self.tiempo_prediccion_total += float(tiempo_segundos)

    def registrar_reinicio(self):
        self.reinicios += 1

    def registrar_resultado_final(self, mejor_fitness, mejor_error=None):
        self.mejor_fitness = float(mejor_fitness)
        self.mejor_error = None if mejor_error is None else float(mejor_error)

    def registrar_evento(self, tipo: str, **payload):
        """
        Registra un evento ligero para depuracion.

        En ejecuciones largas conviene usarlo solo para eventos relevantes:
        entrenamientos, reinicios o cambios de fase.
        """
        evento = {"tipo": str(tipo)}
        evento.update(payload)
        self.eventos.append(evento)

    def registrar_decision_subrogado(self, **payload):
        """
        Registra una decision tomada con el subrogado.

        Esta traza permite analizar el margen fitness_pred - fitness_ref y
        detectar si el filtro esta rechazando candidatos por poco margen.
        """
        self.decisiones_subrogado.append(dict(payload))
        
    @property
    def porcentaje_rechazo(self):
        if self.candidatos_con_subrogado == 0:
            return 0.0
        return self.candidatos_rechazados / self.candidatos_con_subrogado

    @property
    def tiempo_online_total(self):
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
            "entrenamientos_rbf": int(self.entrenamientos_rbf),
            "tiempo_entrenamiento_total": float(self.tiempo_entrenamiento_total),
            "tiempo_prediccion_total": float(self.tiempo_prediccion_total),
            "tiempo_online_total": float(self.tiempo_online_total),
            "reinicios": int(self.reinicios),
            "mejor_fitness": self.mejor_fitness,
            "mejor_error": self.mejor_error,
        }


def guardar_decisiones_subrogado_csv(ruta_base, decisiones):
    decisiones = list(decisiones or [])
    if not decisiones:
        return None

    ruta_csv = Path(ruta_base) / "decisiones_subrogado.csv"
    fieldnames = [
        "evals_reales",
        "generacion",
        "reinicios",
        "evals_desde_reinicio",
        "fitness_pred",
        "fitness_ref",
        "margen_pred_ref",
        "debe_evaluar",
        "motivo",
    ]
    escribir_csv_dicts(ruta_csv, decisiones, fieldnames=fieldnames)
    return str(ruta_csv)
