"""
Métricas de evaluación para modelos subrogados.

Calcula MAE, NMAE, RMSE, NRMSE y correlación de Spearman. Estas métricas
se usan en el benchmark offline y en el resumen final de cada run.
"""

import numpy as np
from scipy.stats import rankdata, pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

METRICAS_MAXIMIZAR = {"spearman"}
METRICAS_MINIMIZAR = {"mae", "nmae", "rmse", "nrmse"}

def calcular_errores_por_muestra(y_true, y_pred):
    """
    Calcula el error absoluto y porcentual por muestra.

    y_true: vector con los valores reales de fitness.
    y_pred: vector con los valores predichos.

    Retorna el error absoluto y porcentual por muestra.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    error_abs = np.abs(y_true - y_pred)
    denominador_pct = np.where(np.abs(y_true) > 1e-12, np.abs(y_true), np.nan)
    error_pct = (error_abs / denominador_pct) * 100.0

    return error_abs, error_pct

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
