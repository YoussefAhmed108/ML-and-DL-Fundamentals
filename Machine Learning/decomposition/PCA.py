import numpy as np

class PCA():
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None


    def fit(self , X):
        number_of_samples , number_of_features = X.shape

        # Step 1: Standardize the data (center it)
        self.mean = np.mean(X , axis=0)
        X_centered = X - self.mean

        # Step 2: Compute the covariance matrix
        C = (X_centered.T @ X_centered) / (number_of_samples - 1)

        # Step 3: Compute eigenvalues and eigenvectors
        eigenvalues , eigenvectors = np.linalg.eig(C)

        # Step 4: Sort eigenvectors by eigenvalues (descending order)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Step 5: Select the first `n_components` eigenvectors
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X):
        # Step 6: Project data onto principal components
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)
    

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
