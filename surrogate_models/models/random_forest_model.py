import numpy as np
from sklearn.ensemble import RandomForestRegressor

from surrogate_models.base import BaseSurrogateModel

class RandomForest(BaseSurrogateModel):
    nombre = "random_forest"
    
    def __init__(self, n_estimators = 200, max_depth = 16, min_samples_leaf = 5, max_features = 0.5, random_state = 42, n_jobs = -1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, min_samples_leaf=min_samples_leaf, n_jobs=n_jobs, max_features=max_features)

    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
    
    def get_params(self):
        return self.model.get_params()