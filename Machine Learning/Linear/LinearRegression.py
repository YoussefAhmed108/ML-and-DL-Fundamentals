import numpy as np
from utils.RegressionMetrics import r2_score
class LinearRegression:
    def __init__(self,fit_intercept=True , positive=False):
        self.fit_intercept=fit_intercept
        self.positive=positive
        self.coef_=None
        self.intercept_=None

    def fit(self, X , y):
        # Add intercept to X if fit_intercept is True (Bias absorption)
        if self.fit_intercept:
            X=np.c_[np.ones(X.shape[0]),X]
        
        # Calculate the coefficients
        # Calculate the pseudo-inverse of X
        X_inv = np.linalg.pinv(X.T @ X) @ X.T
        # Multiply the pseudo-inverse of X with y
        self.coef = X_inv @ y
        self.intercept_= self.coef[-1] if self.fit_intercept else 0
        self.coef_= self.coef[:-1] if self.fit_intercept else self.coef

    def predict(self , X):
        # Add intercept to X if fit_intercept is True (Bias absorption)
        if self.fit_intercept:
            X=np.c_[np.ones(X.shape[0]),X]
        
        # Return the dot product of X and the coefficients
        return X @ self.coef_


    def score(self , X , y):
        predicted_value = self.predict(X)
        # Calculate the R^2 score
        r2 = r2_score(y, predicted_value)

