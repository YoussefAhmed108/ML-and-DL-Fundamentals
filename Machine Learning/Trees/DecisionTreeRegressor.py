import numpy as np
from DecisionTree import DecisionTree
from utils.RegressionMetrics import mean_squared_error

class DecisionTreeRegressor(DecisionTree):
    def __init__(self, max_depth=None , criterion='mse' , min_samples_split=2 , min_samples_leaf=1):
        super().__init__('regression' , max_depth , criterion , min_samples_split , min_samples_leaf , metric=mean_squared_error)

    def fit(self, X, y):
        super().fit(X, y)
    
    def predict(self, X):
        return super().predict(X)
    
    def score(self, X, y):
        return super().score(X, y)