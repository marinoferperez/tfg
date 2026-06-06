from abc import ABC, abstractmethod

class BaseSurrogateModel(ABC):
    nombre = "base"
    
    @abstractmethod
    def fit(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
    
    @abstractmethod
    def get_params(self):
        return {}
    
    # @abstractmethod
    # def score(self, X, y):
    #     """
    #     El método 'score' se utiliza para evaluar el rendimiento del modelo ajustado, generalmente devolviendo una métrica de precisión, error o similar, usando los datos de entrada X y las etiquetas reales y.
    #     """
    #     pass
