"""Modelo subrogado basado en Support Vector Regression (sklearn)."""

from sklearn.svm import SVR as SklearnSVR

from src.surrogates.base import BaseSurrogateModel

class SVR(BaseSurrogateModel):
    """SVR con kernel RBF vía sklearn.svm.SVR."""

    nombre = "svr"

    def __init__(self, kernel="rbf", C=1, epsilon=0.1, gamma="scale"):
        """Parámetros pasados directamente a sklearn.svm.SVR."""
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.model = SklearnSVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
        
    def fit(self, X, y):
        """Ajusta la SVR sobre (X, y)."""
        self.model.fit(X, y)

    def predict(self, X):
        """Predice valores en X."""
        return self.model.predict(X)

    def get_params(self):
        """Devuelve los hiperparámetros del modelo."""
        return self.model.get_params()
