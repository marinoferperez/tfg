"""
Clase base abstracta para los modelos subrogados del proyecto.

Define la interfaz fit/predict/get_params que deben implementar todos los
modelos concretos (RBF, SVR, MLP, LASSO, RandomForest, HGB, XGBoost, …).
"""

from abc import ABC, abstractmethod

class BaseSurrogateModel(ABC):
    """
    Clase base abstracta para los modelos subrogados utilizados.

    Define la interfaz común que deben implementar todos los modelos.
    """
    nombre = "base"
    
    @abstractmethod
    def fit(self, X, y):
        """
        Ajusta el modelo sustituto a partir de un conjunto de datos.

        X: matriz de características o soluciones evaluadas.
        y: vector de valores objetivo o fitness asociados a cada solución.
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        Predice el valor objetivo para nuevas soluciones.

        X: soluciones sobre las que se desea estimar el valor de fitness.

        Retorna las predicciones generadas por el modelo.
        """
        pass
    
    @abstractmethod
    def get_params(self):
        """
        Devuelve los parámetros principales del modelo.

        Este método permite registrar o consultar la configuración utilizada
        por cada modelo sustituto concreto.

        Retorna un diccionario con los parámetros del modelo.
        """
        return {}
