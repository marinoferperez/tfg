"""
Fábrica de modelos subrogados.

Punto de entrada único para instanciar cualquier modelo del proyecto dado su
nombre de cadena. Todos los imports son diferidos para evitar dependencias
circulares y cargas innecesarias de librerías.
"""

MODELOS = ("rbf", "svr", "mlp", "rsm", "random_forest", "hgb", "lasso", "xgboost")


def select_model(model_name, **kwargs):
    """
    Selecciona e instancia un modelo subrogado a partir de su nombre.

    model_name: nombre identificador del modelo que se desea utilizar.
    kwargs: parámetros adicionales que se pasan al constructor del modelo.

    Retorna una instancia del modelo subrogado seleccionado.
    """
    if model_name == "rbf":
        from src.surrogates.models.rbf_model import RBF
        return RBF(**kwargs)
    elif model_name == "svr":
        from src.surrogates.models.svr_model import SVR
        return SVR(**kwargs)
    elif model_name == "mlp":
        from src.surrogates.models.mlp_model import MLP
        return MLP(**kwargs)
    elif model_name == "rsm":
        from src.surrogates.models.rsm_model import RSM
        return RSM(**kwargs)
    elif model_name == "lasso":
        from src.surrogates.models.lasso_model import Lasso
        return Lasso(**kwargs)
    elif model_name == "random_forest":
        from src.surrogates.models.random_forest_model import RandomForest
        return RandomForest(**kwargs)
    elif model_name == "hgb":
        from src.surrogates.models.hgb_model import HGB
        return HGB(**kwargs)
    elif model_name == "xgboost":
        from src.surrogates.models.xgboost_model import XGBoost
        return XGBoost(**kwargs)

    raise ValueError(f"Model name {model_name} not found")
