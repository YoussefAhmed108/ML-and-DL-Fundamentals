import numpy as np
import pandas as pd
from utils.ClassificationMetrics import accuracy_score
from Trees.DecisionTreeClassifier import DecisionTreeClassifier

class GradientBoostingClassifier():
    def __init__(self , loss='log_loss', learning_rate=0.01 , n_estimators=100 , max_depth=3 , subsample=1.0,max_features=None,alpha=0.9,random_state=None,criterion='friedman_mse' ,tol=1e-4 , min_samples_split=2, min_samples_leaf=1 , min_impurity_decrease=0.0):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []
        self.subsample = subsample
        self.max_features = max_features
        self.alpha = alpha
        self.random_state = random_state
        self.criterion = criterion
        self.loss = None
        self.tol = tol
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.classes = None
        self.loss = loss
        self.F_m = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        y_one_hot = np.eye(n_classes)[y.astype(int)]
        class_priors = np.bincount(y) / len(y)
        f_0 = np.log(class_priors)
        self.F_m = np.full((X.shape[0], n_classes), f_0)
        for _ in range(self.n_estimators):
            p = np.exp(self.F_m) / np.sum(np.exp(self.F_m), axis=1, keepdims=True)  # Softmax probabilities
            r_im = y_one_hot - p  # Pseudo residuals
            tree = DecisionTreeClassifier(self.max_depth, self.criterion, self.min_samples_split, self.min_samples_leaf)
            tree.fit(X, r_im)
            self.trees.append(tree)
            f_0 = f_0 + self.learning_rate * tree.predict(X)

            # leaf_nodes = tree.get_leaf_nodes()
            # for node in leaf_nodes:
            #     indices = node.indices  # Indices of samples in this leaf
            #     residuals_in_leaf = r_im[indices]
            #     hessian_in_leaf = p[indices] * (1 - p[indices])  # Second derivative
                
            #     # Compute optimal leaf value
            #     gamma_j = np.sum(residuals_in_leaf) / np.sum(hessian_in_leaf)

            #     # Assign terminal node value
            #     node.value = gamma_j

        predictions = tree.predict(X)
        F_m += self.learning_rate * predictions

    def predict(self, X):
         
        n_samples = X.shape[0]
        n_classes = len(self.classes)

        # Step 1: Initialize with F0 (Log Priors)
        F_m = np.full((n_samples, n_classes), self.f_0)  # Initial prediction

        for tree in self.trees:
            predictions = tree.predict(X)
            F_m += self.learning_rate * predictions

        # Step 3: Convert Raw Scores to Probabilities
        if n_classes == 2:  # Binary classification
            probs = 1 / (1 + np.exp(-F_m[:, 1]))  # Sigmoid function
            preds = (probs > 0.5).astype(int)  # Assign class 1 if probability > 0.5

        else:  # Multi-class classification
            exp_F = np.exp(F_m)
            probs = exp_F / np.sum(exp_F, axis=1, keepdims=True)  # Softmax function
            preds = np.argmax(probs, axis=1)  # Class with highest probability
        
        return preds

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
        