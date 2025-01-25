import numpy as np
class RidgeRegression():
    def __init__(self , alpha=1.0 , fit_intercept=True , max_iter=100 , learning_rate=0.01 , tol=1e-4):
        self.alpha=alpha
        self.fit_intercept=fit_intercept
        self.coef_=None
        self.intercept_=None
    

    def fit(self , X , y):
        n , m = X.shape
        self.coef_ = np.random.randn(m)
        for _ in range(self.max_iter):
            y_pred = X @ self.coef_ + self.intercept_
           