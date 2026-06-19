"""Modelo subrogado basado en Radial Basis Function Interpolation (scipy)."""

from scipy.interpolate import RBFInterpolator

from src.surrogates.base import BaseSurrogateModel

class RBF(BaseSurrogateModel):
    """Interpolador RBF usando scipy.interpolate.RBFInterpolator."""

    nombre = "rbf"

    def __init__(self, neighbors=25, smoothing=0.001, kernel="multiquadric", degree=-1, epsilon=0.1):
        """Parámetros pasados directamente a scipy.interpolate.RBFInterpolator."""
        self.neighbors = neighbors
        self.smoothing = smoothing
        self.kernel = kernel
        self.degree = degree
        self.epsilon = epsilon
        self.model = None

    def fit(self, X, y):
        """Ajusta el interpolador RBF sobre los puntos (X, y)."""
        self.model = RBFInterpolator(
            X,
            y,
            neighbors=self.neighbors,
            smoothing=self.smoothing,
            kernel=self.kernel,
            degree=self.degree,
            epsilon=self.epsilon,
        )

    def predict(self, X):
        """Evalúa el interpolador en los puntos X."""
        if self.model is None:
            raise RuntimeError("El modelo RBF debe entrenarse antes de predecir.")
        return self.model(X)

    def get_params(self):
        """Devuelve los hiperparámetros del modelo."""
        return {
            "neighbors": self.neighbors,
            "smoothing": self.smoothing,
            "kernel": self.kernel,
            "degree": self.degree,
            "epsilon": self.epsilon,
        }
