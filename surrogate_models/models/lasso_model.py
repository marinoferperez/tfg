from sklearn.linear_model import Lasso as SklearnLasso
from surrogate_models.base import BaseSurrogateModel

class Lasso(BaseSurrogateModel):
    """
    Lasso es un modelo de regresión lineal que utiliza la regularización L1 
    para seleccionar las características más importantes.
    """
    nombre = "lasso"

    def __init__(self, alpha=0.01, fit_intercept=True,max_iter=5000,tol=1e-4,random_state=42,selection="cyclic"):
        """
        Inicializa el modelo Lasso con los parámetros por defecto.

        alpha: parámetro de regularización L1.
        fit_intercept: indica si se debe calcular el intercepto.
        max_iter: número máximo de iteraciones.
        tol: tolerancia para el criterio de convergencia.
        random_state: semilla para la reproducibilidad.
        selection: método de selección de características.
        """ 
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.selection = selection

        self.model = SklearnLasso(
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            selection=selection,
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self):
        return self.model.get_params()
