import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from surrogate_models.base import BaseSurrogateModel

class RSM(BaseSurrogateModel):
    nombre = "rsm"
    
    def __init__(self, degree = 2):
        self.degree = degree
        self.model = Pipeline([
            ("poly", PolynomialFeatures(degree=degree)),
            ("linreg", LinearRegression())
        ])
        
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_params(self):
        return {"degree": self.degree}