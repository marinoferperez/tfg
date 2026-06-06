import numpy as np

from smt.surrogate_models import KRG
from surrogate_models.base import BaseSurrogateModel

class Kriging(BaseSurrogateModel):
    nombre = "kriging"

    def __init__(self, corr="matern52", poly="constant"):
        self.corr = corr
        self.poly = poly
        self.model = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        self.model = KRG(
            corr=self.corr,
            poly=self.poly,
            theta0=[1e-2] * X.shape[1],
            print_global=False,
        )
        self.model.set_training_values(X, y)
        self.model.train()

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("El modelo Kriging debe entrenarse antes de predecir.")
        X = np.asarray(X, dtype=float)
        return self.model.predict_values(X).ravel()

    def get_params(self):
        return {"corr": self.corr, "poly": self.poly}
