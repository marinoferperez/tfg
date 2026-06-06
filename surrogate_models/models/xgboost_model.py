from surrogate_models.base import BaseSurrogateModel

try:
    from xgboost import XGBRegressor
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "No se pudo importar 'xgboost'. Instala la dependencia con: "
        "python3 -m pip install --user --break-system-packages xgboost"
    ) from exc


class XGBoost(BaseSurrogateModel):
    nombre = "xgboost"

    def __init__(
        self,
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=1,
        gamma=0.0,
        random_state=42,
        n_jobs=-1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            gamma=gamma,
            random_state=random_state,
            n_jobs=n_jobs,
            objective="reg:squarederror",
            tree_method="hist",
            verbosity=0,
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self):
        return self.model.get_params()