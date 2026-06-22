"""
Selección de modelos subrogados.

Instancia cualquier modelo del proyecto dado su
nombre de cadena. 
"""

MODELOS = ("rbf", "svr", "mlp", "rsm", "random_forest", "hgb", "lasso", "xgboost")

def select_model(nombre_subrogado, **kwargs):
    """
    Selecciona e instancia un modelo subrogado a partir de su nombre.

    nombre_subrogado: nombre del modelo a instanciar; debe ser uno de los valores en MODELOS.
    kwargs: parámetros adicionales que se pasan al constructor del modelo.

    Retorna una instancia del modelo subrogado seleccionado.
    """
    if nombre_subrogado == "rbf":
        from src.surrogates.models.rbf_model import RBF
        return RBF(**kwargs)
    elif nombre_subrogado == "svr":
        from src.surrogates.models.svr_model import SVR
        return SVR(**kwargs)
    elif nombre_subrogado == "mlp":
        from src.surrogates.models.mlp_model import MLP
        return MLP(**kwargs)
    elif nombre_subrogado == "rsm":
        from src.surrogates.models.rsm_model import RSM
        return RSM(**kwargs)
    elif nombre_subrogado == "lasso":
        from src.surrogates.models.lasso_model import Lasso
        return Lasso(**kwargs)
    elif nombre_subrogado == "random_forest":
        from src.surrogates.models.random_forest_model import RandomForest
        return RandomForest(**kwargs)
    elif nombre_subrogado == "hgb":
        from src.surrogates.models.hgb_model import HGB
        return HGB(**kwargs)
    elif nombre_subrogado == "xgboost":
        from src.surrogates.models.xgboost_model import XGBoost
        return XGBoost(**kwargs)

    raise ValueError(f"Subrogado {nombre_subrogado} no encontrado")
