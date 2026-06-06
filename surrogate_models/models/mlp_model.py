from sklearn.neural_network import MLPRegressor

from surrogate_models.base import BaseSurrogateModel

# hidden_layer_sizes=(128, 64)
# activation="tanh"
# alpha=1e-3
# max_iter=2000


# hidden_layer_sizes=(64, 64)
# activation="relu"
# alpha=1e-3
# max_iter=2000

# añadir en memoria:

# buscamos una configuracion de hiperparametros para MLP donde la arquitectura inicial no puede estar imponiendo una compresión demasiado brusca de la representación interna, lo que limita la capacidad del modelo para aproximar funciones continuas con interacciones no lineales complejas. Por ello, se exploran variantes con una capa oculta adicional y mayor anchura intermedia.

class MLP(BaseSurrogateModel):
    nombre = "mlp"

    def __init__(
        self,
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        max_iter=2000,
        random_state=42,
        early_stopping=True
    ):
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
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self):
        return self.model.get_params()
