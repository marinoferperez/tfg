"""Modelo subrogado basado en Support Vector Regression (sklearn)."""

from sklearn.svm import SVR as SklearnSVR

from src.surrogates.base import BaseSurrogateModel


class SVR(BaseSurrogateModel):
    """
    SVR ajusta una función de regresión maximizando el margen y permitiendo
    un error controlado dentro de una banda de tolerancia.
    """

    nombre = "svr"

    def __init__(self, kernel="rbf", C=1, epsilon=0.1, gamma="scale"):
        """
        Inicializa el modelo SVR con los parámetros por defecto.

        kernel: tipo de kernel ("rbf", "linear", "poly", "sigmoid").
        C: parámetro de regularización; valores altos penalizan más los errores de entrenamiento.
        epsilon: margen de tolerancia sin penalización en la función de pérdida epsilon-insensible.
        gamma: coeficiente del kernel; "scale" usa 1/(n_features * X.var()).
        """
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.model = SklearnSVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)

    def fit(self, X, y):
        """
        Ajusta la SVR sobre los datos de entrenamiento.

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
