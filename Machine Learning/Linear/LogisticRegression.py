class LogisticRegression():
    def __init__(self , self.fit_intercept=True, C=1.0 , penalty='l2' , max_iter=100 , learning_rate=0.01 , tol=1e-4):
        self.fit_intercept=fit_intercept
        self.C=C
        self.penalty=penalty
        self.max_iter=max_iter
        self.learning_rate=learning_rate
        self.coef_=None
        self.intercept_=None
        self.tol=tol

    def fit(self, X , y):

        _ , n_features = X.shape
        num_classes = len(np.unique(y))

        self.coef_ = np.random.randn((num_classes, n_features))
        self.intercept_ = np.random.randn(num_classes)

        for _ in range(self.max_iter):
            z = X @ self.coef_.T + self.intercept_
            y_probabilites = softmax(z)
            error = y_probabilites - np.eye(n_classes)[y.astype(int)]
            gradient = X.T @ error
            gradient = regularize_gradient(gradient)
            self.coef_ -= self.learning_rate * gradient
            self.intercept_ -= self.learning_rate * np.sum(error , axis=0)
            if np.linalg.norm(gradient) < self.tol:
                break

   def regularize_gradient(self, gradient):
    """
    Returns the gradient plus the appropriate penalty term
    for L1, L2, or Elastic Net regularization.
    
    Parameters
    ----------
    gradient : np.ndarray
        The unregularized gradient dJ/dw from your loss function.
    
    Returns
    -------
    np.ndarray
        The regularized gradient, i.e. gradient + penalty_derivative.
    """
    if self.penalty == 'l2':
        # L2 adds λ * w to the gradient
        return gradient + (1 / self.C) * self.coef_
    
    elif self.penalty == 'l1':
        # L1 adds λ * sign(w) to the gradient
        return gradient + (1 / self.C) * np.sign(self.coef_)
    
    elif self.penalty == 'elasticnet':
        # Elastic Net is a combination of L1 and L2;
        l2_term = self.coef_ / self.C       
        l1_term = np.sign(self.coef_) / self.C  
        return gradient + l1_term + l2_term
    
    else:
        # No regularization
        return gradient


    def predict(self , X):
        probailities = self.predict_proba(X)
        return np.argmax(probailities , axis=1)


    def predict_proba(self , X):
        z = X @ self.coef_.T + self.intercept_
        return softmax(z)

    def score(self , X , y):
        y_pred = self.predict(X)
        # Calculate the accuracy
        return np.mean(y_pred == y)

