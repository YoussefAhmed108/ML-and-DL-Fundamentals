import numpy as np
from utils.ClassificationMetrics import accuracy_score
from DecisionTree import DecisionTree

class DecisionTreeClasssifier(DecisionTree):
    def __init__(self, max_depth=None , criterion='gini' , min_samples_split=2 , min_samples_leaf=1):
        super().__init__('classification' , max_depth , criterion , min_samples_split , min_samples_leaf , metric=accuracy_score )

    def fit(self, X, y):
        super().fit(X, y)
    
    def predict(self, X):
        return super().predict(X)
    
    def score(self, X, y):
        return super().score(X, y)
    