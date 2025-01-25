import numpy as np
from utils.RegressionMetrics import mean_squared_error
class KNNRegressor():
    def __init__(self, n_neighbours=3):
        self.n_neighbours = n_neighbours

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y_pred[i] = np.mean(self.y[np.argsort(np.linalg.norm(self.X - X[i], axis=1))[:self.n_neighbours]])
        return y_pred

    def score(self, X, y):
        return mean_squared_error(y, self.predict(X))