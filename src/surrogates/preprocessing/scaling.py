"""
Utilidades de escalado de variables para modelos subrogados.

El dominio de CEC2017 es [-100, 100] en todas las dimensiones. El escalado
de X es fijo (no se ajusta sobre datos) y se aplica antes de entrenar o
predecir con cualquier modelo. El escalado de Y sí se ajusta sobre los datos
de entrenamiento y se invierte sobre las predicciones.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.benchmark.cec2017_problem import _LIMITE_SUP


# límite del dominio CEC2017: [-100, 100] → se normaliza a [-1, 1].
DOMAIN_BOUND = _LIMITE_SUP

# modelos basados en árboles que no requieren escalado de Y.
MODELOS_ARBOL = {"random_forest", "hgb", "xgboost"}


def escalar_X(X):
    """
    Escala las variables de entrada al rango [-1, 1] dividiendo por DOMAIN_BOUND.

    La transformación es fija: no depende de los datos de entrenamiento y puede
    aplicarse igual en train, validación y predicción en línea.

    X: array de forma (n_muestras, n_dimensiones) en el dominio [-100, 100].
    """
    return np.asarray(X, dtype=float) / DOMAIN_BOUND


def construir_escalador_y(nombre_subrogado):
    """
    Devuelve un StandardScaler si el modelo requiere escalado de Y, o None.

    Los modelos de árbol (random_forest, hgb, xgboost) son invariantes a escala,
    por lo que no necesitan escalado de Y.

    nombre_subrogado: nombre del modelo subrogado.
    """
    if nombre_subrogado in MODELOS_ARBOL:
        return None
    return StandardScaler()


def ajustar_y(scaler, y_train):
    """
    Ajusta el escalador sobre y_train y devuelve y_train escalado.

    scaler: escalador de Y devuelto por construir_escalador_y; si es None,
            devuelve y_train sin modificar.
    y_train: valores de fitness de entrenamiento.
    """
    if scaler is None:
        return y_train
    
    return scaler.fit_transform(y_train.reshape(-1, 1)).ravel()


def invertir_y(scaler, y_pred):
    """
    Invierte el escalado de Y sobre las predicciones del modelo.

    scaler: escalador ajustado; si es None, devuelve y_pred sin modificar.
    y_pred: predicciones en el espacio escalado.
    """
    if scaler is None:
        return y_pred
    return scaler.inverse_transform(np.asarray(y_pred).reshape(-1, 1)).ravel()
