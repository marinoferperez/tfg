"""Modelo subrogado basado en Random Forest Regressor (sklearn)."""

from sklearn.ensemble import RandomForestRegressor

from src.surrogates.base import BaseSurrogateModel


class RandomForest(BaseSurrogateModel):
    """
    Random Forest combina varios árboles de decisión y promedia sus predicciones
    para reducir la varianza del modelo.
    """

    nombre = "random_forest"

    def __init__(self, n_estimators=200, max_depth=16, min_samples_leaf=5, max_features=0.5, random_state=42, n_jobs=-1):
        """
        Inicializa el modelo Random Forest con los parámetros por defecto.

        n_estimators: número de árboles en el bosque.
        max_depth: profundidad máxima de cada árbol.
        min_samples_leaf: número mínimo de muestras requeridas en un nodo hoja.
        max_features: fracción de features consideradas en cada split.
        random_state: semilla para reproducibilidad.
        n_jobs: número de trabajos en paralelo; -1 usa todos los núcleos disponibles.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=random_state, min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs, max_features=max_features
        )

    def fit(self, X, y):
        """
        Ajusta el Random Forest sobre los datos de entrenamiento.

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
        return self.model.get_params()
