"""Modelo subrogado basado en Radial Basis Function Interpolation (scipy)."""

from scipy.interpolate import RBFInterpolator

from src.surrogates.base import BaseSurrogateModel


class RBF(BaseSurrogateModel):
    """
    RBF aproxima la función objetivo mediante una combinación de funciones de
    base radial centradas en soluciones ya evaluadas.
    """

    nombre = "rbf"

    def __init__(self, neighbors=50, smoothing=1, kernel="gaussian", degree=-1, epsilon=1.0):
        """
        Inicializa el modelo RBF con los parámetros por defecto.

        neighbors: número de vecinos más cercanos usados en la interpolación; None usa todos.
        smoothing: factor de suavizado; 0 produce interpolación exacta.
        kernel: función de base radial ("multiquadric", "thin_plate_spline", "gaussian", etc.).
        degree: grado del polinomio de soporte; -1 deja que scipy elija el mínimo requerido.
        epsilon: escala de forma usada por kernels como "multiquadric" o "inverse_quadratic".
        """
        self.neighbors = neighbors
        self.smoothing = smoothing
        self.kernel = kernel
        self.degree = degree
        self.epsilon = epsilon
        # RBFInterpolator necesita X e y, por lo que se construye en fit.
        self.model = None

    def fit(self, X, y):
        """
        Ajusta el interpolador RBF sobre los puntos de entrenamiento.

        X: matriz de características (n, d).
        y: valores objetivo de fitness.
        """
        self.model = RBFInterpolator(
            X, y, neighbors=self.neighbors, smoothing=self.smoothing,
            kernel=self.kernel, degree=self.degree, epsilon=self.epsilon
        )

    def predict(self, X):
        """
        Predice el fitness para nuevas soluciones.

        X: soluciones sobre las que estimar el fitness.
        """
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
