import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from surrogate_models.base import BaseSurrogateModel

class HGB(BaseSurrogateModel):
    nombre = "hit_gradient_boosting"
    
    def __init__(self, max_iter=200, learning_rate=0.05, random_state=42, max_depth=4,
                 l2_regularization=1e-3, early_stopping="auto", validation_fraction=0.1):
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.l2_regularization = l2_regularization
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.model = HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=learning_rate,
            random_state=random_state,
            max_depth=max_depth,
            l2_regularization=l2_regularization,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
        )
        
    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
    
    def get_params(self):
        return self.model.get_params()