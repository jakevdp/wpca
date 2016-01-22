import numpy as np

from .wpca import pca, wpca_delchambre
from sklearn.base import BaseEstimator, TransformerMixin

class PCA(BaseEstimator, TransformerMixin):
    """Principal Component Analysis

    This is a standard Principal Component Analysis implementation
    based on the Singular Value Decomposition.

    Parameters
    ----------
    TODO

    Attributes
    ----------
    mean_ :
    components_ :
    explained_variance_ :
    """
    def __init__(self, n_components=None):
        self.n_components = n_components

    @staticmethod
    def _compute_mean(X):
        return X.mean(0)

    def fit_transform(self, X):
        if self.n_components is None:
            n_components = X.shape[1]
        else:
            n_components = self.n_components
        self.mean_ = self._compute_mean(X)
        P, C, sigma, var_tot = pca((X - self.mean_).T, n_components)
        self.components_ = P.T
        self.explained_variance_ = sigma
        self.explained_variance_ratio_ = sigma / var_tot
        return C.T

    def fit(self, X):
        self.fit_transform(X)
        return self

    def transform(self, X):
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, X):
        return self.mean_ + X @ self.components_


class WPCA(BaseEstimator, TransformerMixin):
    """Weighted Principal Component Analysis

    This is a direct implementation of weighted PCA based on the eigenvalue
    decomposition of the weighted covariance matrix following
    Delchambre (2014) [1]_.

    Parameters
    ----------
    TODO

    Attributes
    ----------
    mean_ :
    components_ :
    explained_variance_ :

    References
    ----------
    .. [1] Delchambre, L. MNRAS 2014 446 (2): 3545-3555 (2014)
           http://arxiv.org/abs/1412.4533
    """
    def __init__(self, n_components=None, xi=0):
        self.n_components = n_components
        self.xi = xi

    @staticmethod
    def _compute_mean(X, weights):
        if weights is None or np.isscalar(weights):
            return X.mean(0)
        else:
            assert X.shape == weights.shape
            return (X * weights).sum(0) / weights.sum(0)

    def _fit(self, X, weights=None, compute_transform=True):
        if self.n_components is None:
            n_components = X.shape[1]
        else:
            n_components = self.n_components

        W = weights.T if weights is not None else weights
        self.mean_ = self._compute_mean(X, weights)
        P, C, sigma, var_tot =\
            wpca_delchambre((X - self.mean_).T,
                            n_components, W=W, xi=self.xi,
                            compute_transform=compute_transform)
        self.components_ = P.T
        self.explained_variance_ = sigma

        # XXX: a weighted covariance matrix is not necessarily
        # positive-semidefinite, so the variance ratio can rise above 1.
        # I have no idea what this means, or whether it's a useful concept
        # in this generalization.
        self.explained_variance_ratio_ = sigma / var_tot

        if compute_transform:
            return C.T

    def fit_transform(self, X, weights=None):
        return self._fit(X, weights, compute_transform=True)


    def fit(self, X, weights=None):
        self._fit(X, weights, compute_transform=False)
        return self

    def transform(self, X, weights=None):
        Xtrans = np.zeros((X.shape[0], self.n_components))
        Xmu = X - self.mean_
        P = self.components_
        if weights is None:
            Xtrans = Xmu @ P.T
        else:
            for i in range(X.shape[0]):
                W2 = np.diag(weights[i] ** 2)
                Xtrans[i] = np.linalg.solve(P @ W2 @ P.T, P @ W2 @ Xmu[i])
        return Xtrans

    def inverse_transform(self, X):
        return self.mean_ + X @ self.components_
