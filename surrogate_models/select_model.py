def select_model(model_name, **kwargs):
    if model_name == "rbf":
        from surrogate_models.models.rbf_model import RBF
        return RBF(**kwargs)
    elif model_name == "svr":
        from surrogate_models.models.svr_model import SVR
        return SVR(**kwargs)
    elif model_name == "mlp":
        from surrogate_models.models.mlp_model import MLP
        return MLP(**kwargs)
    elif model_name == "rsm":
        from surrogate_models.models.rsm_model import RSM
        return RSM(**kwargs)
    elif model_name == "lasso":
        from surrogate_models.models.lasso_model import Lasso
        return Lasso(**kwargs)
    elif model_name == "random_forest":
        from surrogate_models.models.random_forest_model import RandomForest
        return RandomForest(**kwargs)
    elif model_name == "hgb":
        from surrogate_models.models.hgb_model import HGB
        return HGB(**kwargs)
    elif model_name == "xgboost":
        from surrogate_models.models.xgboost_model import XGBoost
        return XGBoost(**kwargs)
    elif model_name == "kriging":
        from surrogate_models.models.kriging_model import Kriging
        return Kriging(**kwargs)
    elif model_name == "pce":
        from surrogate_models.models.pce_model import PCE
        return PCE(**kwargs)

    raise ValueError(f"Model name {model_name} not found")
