from Trees.DecisionTreeClassifier import DecisionTreeClassifier
from utils.ClassificationMetrics import accuracy_score
import numpy as np
from scipy.stats import mode


class RandomForestClassifier():
    def __init__(self, n_estimators=100, max_depth=None, criterion='gini', min_samples_split=2, min_samples_leaf=1 , max_features='sqrt' , bootstrap=True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.max_features = max_features
        self.trees = []


    def fit(self, X, y):
        n_samples, n_features = X.shape

        if isinstance(self.max_features, str):
            if self.max_features == 'sqrt':
                self.max_features = int(np.sqrt(n_features))
            elif self.max_features == 'log2':
                self.max_features = int(np.log2(n_features))
            else:
                self.max_features = n_features  # Use all features
        elif self.max_features is None:
            self.max_features = n_features
        
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(self.max_depth, self.criterion, self.min_samples_split, self.min_samples_leaf)
            # Select random features
            selected_features = np.random.choice(n_features, self.max_features, replace=False)
            if self.bootstrap:
                X_sample , y_sample = self._bootstrap_sample(X, y)
            else:
                X_sample , y_sample = X, y
            
            tree.fit(X_sample[: , selected_features], y_sample)
            self.trees.append((tree, selected_features))
    
    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)  # Sample with replacement
        return X[indices], y[indices]
    
    def predict(self, X):
        predictions = np.array([tree.predict(X[:, features]) for tree, features in self.trees])
        predictions = np.swapaxes(predictions, 0, 1)
        y_pred = mode(predictions, axis=1).mode.flatten()
        return np.array(y_pred)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)