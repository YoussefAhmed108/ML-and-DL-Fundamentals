import numpy as np
from utils.RegressionMetrics import r2_score
class RidgeRegression():
    def __init__(self , alpha=1.0 , fit_intercept=True , max_iter=100 , learning_rate=0.01 , tol=1e-4):
        self.alpha=alpha
        self.fit_intercept=fit_intercept
        self.coef_=None
        self.intercept_=None
    

    def fit(self , X , y):
        n , m = X.shape
        X_with_bias = np.c_[np.ones(n), X] if self.fit_intercept else X
        # Compute coefficients using the closed-form solution
        A = X_with_bias.T @ X_with_bias + (self.alpha * np.eye(X_with_bias.shape[1]))
        b = X_with_bias.T @ y
        self.coef_ = np.linalg.solve(A, b)
        self.intercept_ = self.coef_[0] if self.fit_intercept else 0
        self.coef_ = self.coef_[1:] if self.fit_intercept else self.coef_

    def predict(self , X):
        return X @ self.coef_ + self.intercept_
    
    def score(self , X , y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
           