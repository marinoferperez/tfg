"""Modelo subrogado basado en Multi-Layer Perceptron Regressor (sklearn)."""

from sklearn.neural_network import MLPRegressor

from src.surrogates.base import BaseSurrogateModel


class MLP(BaseSurrogateModel):
    """MLP Regressor con early stopping vía sklearn.neural_network.MLPRegressor."""

    nombre = "mlp"

    def __init__(
        self,
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        max_iter=2000,
        random_state=42,
        early_stopping=True,
    ):
        """Parámetros pasados directamente a sklearn.neural_network.MLPRegressor."""
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=early_stopping,
        )

    def fit(self, X, y):
        """Ajusta el MLP sobre (X, y)."""
        self.model.fit(X, y)

    def predict(self, X):
        """Predice valores en X."""
        return self.model.predict(X)

    def get_params(self):
        """Devuelve los hiperparámetros del modelo."""
        return self.model.get_params()
