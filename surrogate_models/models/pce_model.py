import numpy as np

from smt.surrogate_models import PCE as SMT_PCE
from surrogate_models.base import BaseSurrogateModel

class PCE(BaseSurrogateModel):
    nombre = "pce"

    def __init__(self, order=3):
        self.order = order
        self.model = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        self.model = SMT_PCE(order=self.order)
        self.model.set_training_values(X, y)
        self.model.train()

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("El modelo PCE debe entrenarse antes de predecir.")
        X = np.asarray(X, dtype=float)
        return self.model.predict_values(X).ravel()

    def get_params(self):
        return {"order": self.order}
