from utils.RegressionMetrics import r2_score
from utils.Functions import soft_threshold
import numpy as np
class LassoRegression():
    def __init__(self, alpha=0.1, max_iter=1000, tol=1e-4 , fit_intercept=True):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None
        self.fit_intercept = fit_intercept

    
    def fit(self, X, y):
        n , m = X.shape
        self.coef_ = np.random.randn(m)
        self.intercept_ = 0 if self.fit_intercept else None  # Initialize intercept
        
        for _ in range(self.max_iter):
            coef_prev = self.coef_.copy()  # Save coefficients for convergence check
            for j in range(m):
                X_j = X[:, j]
                y_pred = self.predict(X)
                r_j = y - y_pred + self.coef_[j] * X_j
                # Calculate L2 norm of X_j
                l2_norm_X_j = np.linalg.norm(X_j, 2)

                rho_j = (X_j @ r_j) /  (l2_norm_X_j ** 2)
                gamma = self.alpha / (l2_norm_X_j ** 2)
                w_j_new = soft_threshold(rho_j, gamma)
                self.coef_[j] = w_j_new

            if self.fit_intercept:
                self.intercept_ = np.mean(y - X @ self.coef_)
            

            coef_change = np.linalg.norm(self.coef_ - coef_prev)
            if coef_change < self.tol:
                break
        print("Fit")


    def predict(self, X):
        return X @ self.coef_ + self.intercept_
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)