"""
Politica de decision para el filtro subrogado online.

Decide si una prediccion del modelo justifica gastar una evaluacion real.

Para CEC2017 trabajamos en minimizacion:
    - si f_predicha(candidato) < f_real(referencia), se evalua realmente;
    - en caso contrario, se rechaza sin consumir evaluacion real.
"""

import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class DecisionSubrogado:
    """
    Resultado de aplicar la politica de decision del subrogado.

    debe_evaluar indica si el candidato debe pasar a evaluacion real.
    Si es False, el candidato se rechaza sin consumir evaluacion de la funcion objetivo.
    """
    debe_evaluar: bool
    fitness_pred: float | None
    fitness_ref: float | None
    motivo: str
    
    @property
    def rechazado(self):
        """True si el candidato fue rechazado sin evaluar la función objetivo."""
        return not self.debe_evaluar
    
class PoliticaSubrogado:
    """
    Politica de rechazo para la integracion online.
    """
    
    def __init__(self, minimizacion=True):
        """minimizacion: True para CEC2017 (se busca minimizar fitness)."""
        self.minimizacion = bool(minimizacion)

    def decidir(self, fitness_pred, fitness_ref):
        """
        Decide si un candidato debe evaluarse realmente dado su fitness predicho.

        fitness_pred: predicción del modelo subrogado para el candidato.
        fitness_ref: fitness real del individuo de referencia (padre en DE/SHADE, peor individuo en AGE).

        Retorna DecisionSubrogado con debe_evaluar=True si el candidato es prometedor.
        """
        pred = self._float_valido(fitness_pred)
        ref = self._float_valido(fitness_ref)
        
        # si el valor es invalido, se fuerza evaluacion real. 
        # se prefiere gastar una evaluacion antes que rechazar por un fallo numerico del subrogado.
        if pred is None:
            return DecisionSubrogado(
                debe_evaluar=True,
                fitness_pred=None,
                fitness_ref=ref,
                motivo="prediccion_invalida",
            )
        
        if ref is None:
            return DecisionSubrogado(
                debe_evaluar=True,
                fitness_pred=pred,
                fitness_ref=None,
                motivo="referencia_invalida", 
            )
            
        if self.minimizacion:
            candidato_prometedor = pred < ref
        else: 
            candidato_prometedor = pred > ref
            
        if candidato_prometedor:
            return DecisionSubrogado(
                debe_evaluar=True,
                fitness_pred=pred,
                fitness_ref=ref,
                motivo="aceptado_por_subrogado", 
            )

        return DecisionSubrogado(
            debe_evaluar=False,
            fitness_pred=pred,
            fitness_ref=ref,
            motivo="rechazado_por_subrogado",
        )
        
    @staticmethod
    def _float_valido(valor):
        """
        Convierte un escalar o un array con un unico valor a float.

        valor: predicción o referencia a normalizar (puede ser ndarray de shape (1,) o escalar).

        Devuelve None si el valor no es numerico, no es finito o contiene mas
        de un elemento.
        """
        try:
            array = np.asarray(valor, dtype=float)
            if array.size != 1:
                return None
            escalar = float(array.reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            return None

        if not np.isfinite(escalar):
            return None

        return escalar