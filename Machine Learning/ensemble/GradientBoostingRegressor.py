import numpy as np
from Trees.DecisionTreeRegressor import DecisionTreeRegressor
from utils.RegressionMetrics import r2_score

class GradientBoostingRegressor():
    def __init_(self, n_estimators=100, max_depth=3, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.trees = []
        self.losses = []

    def fit(self, X, y):
        F0 = np.mean(X)
        for i in range(self.n_estimators):
            residuals = y - F0
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)
            F0 += self.learning_rate * tree.predict(X)
            self.losses.append(r2_score(y, F0))


    def predict(self, X):
        F0 = np.mean(X)
        for tree in self.trees:
            F0 += self.learning_rate * tree.predict(X)
        return F0   
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)