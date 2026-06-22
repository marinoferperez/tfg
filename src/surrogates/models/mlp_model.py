"""Modelo subrogado basado en Multi-Layer Perceptron Regressor (sklearn)."""

from sklearn.neural_network import MLPRegressor

from src.surrogates.base import BaseSurrogateModel


class MLP(BaseSurrogateModel):
    """
    MLP es una red neuronal multicapa que aprende una relación no lineal entre
    las soluciones candidatas y su valor de fitness.
    """

    nombre = "mlp"

    def __init__(self, hidden_layer_sizes=(128, 64), activation="relu", solver="adam", alpha=1e-3, max_iter=2000, random_state=42, early_stopping=True):
        """
        Inicializa el modelo MLP con los parámetros por defecto.

        hidden_layer_sizes: tupla con el número de neuronas por capa oculta.
        activation: función de activación de las capas ocultas ("relu", "tanh", "logistic").
        solver: algoritmo de optimización ("adam", "sgd", "lbfgs").
        alpha: término de regularización L2 aplicado sobre los pesos.
        max_iter: número máximo de iteraciones de entrenamiento.
        random_state: semilla para reproducibilidad.
        early_stopping: si True, reserva una fracción de validación para detención anticipada.
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes, activation=activation,
            solver=solver, alpha=alpha, max_iter=max_iter,
            random_state=random_state, early_stopping=early_stopping
        )

    def fit(self, X, y):
        """
        Ajusta el MLP sobre los datos de entrenamiento.

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
