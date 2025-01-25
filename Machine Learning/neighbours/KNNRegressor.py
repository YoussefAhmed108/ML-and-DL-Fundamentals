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
        for x in X:
            distances = np.linalg.norm(self.X - x, axis=1)
            nearest_neighbors = self.y[np.argsort(distances)[:self.n_neighbours]]
            y_pred.append(np.mean(nearest_neighbors))
        return y_pred

    def score(self, X, y):
        return mean_squared_error(y, self.predict(X))