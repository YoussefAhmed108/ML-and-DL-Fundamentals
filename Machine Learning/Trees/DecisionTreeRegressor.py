import numpy as np
from .DecisionTree import DecisionTree
from utils.RegressionMetrics import mean_squared_error, r2_score
from utils.Functions import gini_index
from utils.Functions import entropy

class DecisionTreeRegressor(DecisionTree):
    def __init__(self, max_depth=None , criterion='gini' , min_samples_split=2 , min_samples_leaf=1):
        criterion = gini_index if criterion == 'gini' else entropy
        super().__init__('regression' , max_depth , criterion , min_samples_split , min_samples_leaf , metric=r2_score)

    def fit(self, X, y):
        super().fit(X, y)
    
    def predict(self, X):
        return super().predict(X)
    
    def score(self, X, y):
        return super().score(X, y)
    
    def print_tree(self , feature_names):
        super()._show_tree(self.tree , feature_names)