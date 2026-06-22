"""Modelo subrogado basado en Response Surface Methodology (sklearn)."""

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from src.surrogates.base import BaseSurrogateModel


class RSM(BaseSurrogateModel):
    """
    RSM aproxima la función objetivo mediante una regresión lineal sobre
    características polinómicas.
    """

    nombre = "rsm"

    def __init__(self, degree=2):
        """
        Inicializa el modelo RSM con los parámetros por defecto.

        degree: grado del polinomio. Por defecto, superficie cuadrática.
        """
        self.degree = degree
        self.model = Pipeline(
            [
                ("poly", PolynomialFeatures(degree=degree)),
                ("linreg", LinearRegression()),
            ]
        )

    def fit(self, X, y):
        """
        Ajusta el RSM sobre los datos de entrenamiento.

        X: matriz de características (n, d).
        y: valores objetivo de fitness.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predice el fitness para nuevas soluciones.

        X: soluciones sobre las que estimar el fitness.
        """
        return self.model.predict(X)

    def get_params(self):
        """Devuelve los hiperparámetros del modelo."""
        return {"degree": self.degree}
