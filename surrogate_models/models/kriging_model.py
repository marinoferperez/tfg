import numpy as np

from smt.surrogate_models import KRG
from surrogate_models.base import BaseSurrogateModel

class Kriging(BaseSurrogateModel):
    """
    Kriging es un modelo de regresión no paramétrico que utiliza la correlación espacial 
    para aproximar la función objetivo.
    """
    nombre = "kriging"

    def __init__(self, corr="matern52", poly="constant"):
        """
        Inicializa el modelo Kriging con los parámetros por defecto.

        corr: tipo de correlación espacial.
        poly: tipo de polinomio.
        """
        self.corr = corr
        self.poly = poly
        self.model = KRG( corr=corr, poly=poly, theta0=[1e-2] * X.shape[1], print_global=False)
        
    def fit(self, X, y):
        self.model.set_training_values(X, y)
        self.model.train()
        
    def predict(self, X):
        return self.model.predict_values(X)

    def get_params(self):
        return {"corr": self.corr, "poly": self.poly}
