"""
Métricas de evaluación para modelos subrogados.

Calcula MAE, NMAE, RMSE, NRMSE y correlación de Spearman.
"""

import numpy as np
from scipy.stats import rankdata, pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

METRICAS_MAXIMIZAR = {"spearman"}
METRICAS_MINIMIZAR = {"mae", "nmae", "rmse", "nrmse"}

def calcular_metricas(y_true, y_pred):
    """
    Calcula las métricas de evaluación del modelo.

    y_true: vector con los valores reales de fitness.
    y_pred: vector con los valores predichos.

    Retorna un diccionario con las métricas calculadas: RMSE, NRMSE, Spearman, MAE, NMAE.
    """
    ranks_true = rankdata(y_true, method="ordinal")
    ranks_pred = rankdata(y_pred, method="ordinal")
            
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "nmae": float(mean_absolute_error(y_true, y_pred)) / np.mean(np.abs(y_true)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "nrmse": float(np.sqrt(mean_squared_error(y_true, y_pred))) / np.mean(np.abs(y_true)),
        "spearman": float(pearsonr(ranks_true, ranks_pred)[0]),
    }
