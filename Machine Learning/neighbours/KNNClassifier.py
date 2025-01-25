import numpy as np
from utils.ClassificationMetrics import accuracy_score
class KNNClassifier():
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y
    

    def predict(self, X):
        y_pred = []
        for x in X:
            distances = np.linalg.norm(self.X - x, axis=1)
            nearest_neighbors = self.y[np.argsort(distances)[:self.n_neighbors]]
            y_pred.append(np.bincount(nearest_neighbors).argmax())
        return y_pred
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
    
    
