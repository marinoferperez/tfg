from scipy.interpolate import RBFInterpolator

from surrogate_models.base import BaseSurrogateModel

class RBF(BaseSurrogateModel):
    nombre = "rbf"

    ## -- configuración de hiperparámetros por defecto para el analisis del comportamiento de los distintos modelos sobre las estrategias offline
    
    # def __init__(self, neighbors=50, smoothing=1, kernel="gaussian", degree=-1, epsilon=1.0):

    ## -- configuración de hiperparámetros ajustada para la posterior integración online.
    
    def __init__(self, neighbors=25, smoothing=0.001, kernel="multiquadric", degree=-1, epsilon=0.1):
        self.neighbors = neighbors
        self.smoothing = smoothing
        self.kernel = kernel
        self.degree = degree
        self.epsilon = epsilon
        self.model = None

    def fit(self, X, y):
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
        if self.model is None:
            raise RuntimeError("El modelo RBF debe entrenarse antes de predecir.")
        return self.model(X)

    def get_params(self):
        return {
            "neighbors": self.neighbors,
            "smoothing": self.smoothing,
            "kernel": self.kernel,
            "degree": self.degree,
            "epsilon": self.epsilon,
        }
