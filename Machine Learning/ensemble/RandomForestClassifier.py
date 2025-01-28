from Trees.DecisionTreeClassifier import DecisionTreeClassifier
from utils.ClassificationMetrics import accuracy_score
import numpy as np

class RandomForestClassifier():
    def __init__(self, n_estimators=100, max_depth=None, criterion='gini', min_samples_split=2, min_samples_leaf=1 , max_features=3 , bootstrap=True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.max_features = max_features
        self.trees = []


    def fit(self, X, y):
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(self.max_depth, self.criterion, self.min_samples_split, self.min_samples_leaf)
            # Select random features
            random_features = np.random.choice(X.shape[1], self.max_features, replace=False)
            if self.bootstrap:
                X_sample , y_sample = self._bootstrap_sample(X, y)
            else:
                X_sample , y_sample = X, y
            
            tree.fit(X_sample[: , random_features], y_sample)
            self.trees.append(tree)
    
    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]
    
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        predictions = np.swapaxes(predictions, 0, 1)
        y_pred = [np.bincount(prediction).argmax() for prediction in predictions]
        return np.array(y_pred)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)