"""Modelo subrogado basado en Response Surface Methodology (regresión polinómica)."""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from src.surrogates.base import BaseSurrogateModel

class RSM(BaseSurrogateModel):
    """RSM: regresión lineal sobre features polinómicas de grado configurable."""

    nombre = "rsm"

    def __init__(self, degree=2):
        """degree: grado del polinomio. Por defecto, superficie cuadrática."""
        self.degree = degree
        self.model = Pipeline([
            ("poly", PolynomialFeatures(degree=degree)),
            ("linreg", LinearRegression())
        ])
        
    def fit(self, X, y):
        """Ajusta el RSM sobre (X, y)."""
        self.model.fit(X, y)

    def predict(self, X):
        """Predice valores en X."""
        return self.model.predict(X)

    def get_params(self):
        """Devuelve los hiperparámetros del modelo."""
        return {"degree": self.degree}