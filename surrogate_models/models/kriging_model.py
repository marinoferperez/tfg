from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

from surrogate_models.base import BaseSurrogateModel


class Kriging(BaseSurrogateModel):
    """
    Kriging (Proceso Gaussiano) con kernel Matérn 5/2, usado como referencia
    clásica frente a modelos de aprendizaje automático más complejos.
    """
    nombre = "kriging"

    def __init__(self, nu=2.5, length_scale=1.0, n_restarts=3):
        self.nu = nu
        self.length_scale = length_scale
        self.n_restarts = n_restarts
        self.model = None

    def fit(self, X, y):
        kernel = ConstantKernel() * Matern(nu=self.nu, length_scale=self.length_scale)
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=self.n_restarts,
            normalize_y=False,
        )
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self):
        return {
            "nu": self.nu,
            "length_scale": self.length_scale,
            "n_restarts": self.n_restarts,
        }
