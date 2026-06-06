from sklearn.svm import SVR as SklearnSVR

from surrogate_models.base import BaseSurrogateModel

class SVR(BaseSurrogateModel):
    nombre = "svr"
    
    def __init__(self, kernel = "rbf", C = 1, epsilon = 0.1, gamma = "scale"):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.model = SklearnSVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
        
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_params(self):
        return self.model.get_params()
