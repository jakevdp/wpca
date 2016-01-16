from .wpca import pca, wpca_delchambre
from sklearn.base import BaseEstimator, TransformerMixin

class PCA(BaseEstimator, TransformerMixin):
    """PCA implemented with SVD"""
    def __init__(self, n_components=None):
        self.n_components = n_components

    def center_data(self, X):
        return X - X.mean(0)

    def fit_transform(self, X):
        if self.n_components is None:
            n_components = X.shape[1]
        else:
            n_components = self.n_components
        self.mean_ = X.mean(0)
        P, C, sigma = pca((X - self.mean_).T,
                          n_components)
        self.components_ = P.T
        self.explained_variance_ = sigma
        return C.T

    def fit(self, X):
        self.fit_transform(X)
        return self

    def transform(self, X):
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, X):
        return self.mean_ + X @ self.components_
