"""Modelo subrogado basado en Histogram-based Gradient Boosting (sklearn)."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from src.surrogates.base import BaseSurrogateModel

class HGB(BaseSurrogateModel):
    """
    Histogram-based Gradient Boosting (HGB) es un modelo de boosting basado en árboles 
    de decisión.
    """
    nombre = "hit_gradient_boosting"
    
    def __init__(self, max_iter=200, learning_rate=0.05, random_state=42, max_depth=4,
                 l2_regularization=1e-3, early_stopping="auto", validation_fraction=0.1):
        """
        Inicializa el modelo HGB con los parámetros por defecto.

        max_iter: número máximo de iteraciones de boosting.
        learning_rate: tasa de aprendizaje aplicada en cada iteración.
        random_state: semilla utilizada para la reproducibilidad.
        max_depth: profundidad máxima de los árboles base.
        l2_regularization: regularización L2 aplicada durante el entrenamiento.
        early_stopping: criterio de parada temprana.
        validation_fraction: fracción de datos para validación si se aplica early stopping.
        """
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.l2_regularization = l2_regularization
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.model = HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=learning_rate,
            random_state=random_state,
            max_depth=max_depth,
            l2_regularization=l2_regularization,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
        )
        
    def fit(self, X, y):
        """Ajusta el HGB sobre (X, y)."""
        self.model.fit(X, y)

    def predict(self, X):
        """Predice valores en X."""
        return self.model.predict(X)

    def get_params(self):
        """Devuelve los hiperparámetros del modelo."""
        return self.model.get_params()