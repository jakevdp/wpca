import numpy as np
from scipy import linalg

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

    def fit_transform(self, X):
        if self.n_components is None:
            n_components = X.shape[1]
        else:
            n_components = self.n_components

        self.mean_ = X.mean(0)
        U, s, VT = np.linalg.svd(X - self.mean_)
        self.components_ = VT[:self.n_components]
        var = s ** 2 / X.shape[0]
        self.explained_variance_ = var[:self.n_components]
        self.explained_variance_ratio_ = var[:self.n_components] / var.sum()
        return s[:self.n_components] * U[:, :self.n_components]

    def fit(self, X):
        self.fit_transform(X)
        return self

    def transform(self, X):
        return np.dot(X - self.mean_, self.components_.T)

    def inverse_transform(self, X):
        return self.mean_ + np.dot(X, self.components_)

    def reconstruct(self, X):
        return self.inverse_transform(self.transform(X))


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
    def __init__(self, n_components=None, xi=0, regularize=False):
        self.n_components = n_components
        self.xi = xi
        self.regularize = regularize

    def fit(self, X, weights=None):
        if self.n_components is None:
            n_components = min(X.shape)
        else:
            n_components = self.n_components

        if weights is None:
            self.mean_ = X.mean(0)
            weights = np.ones_like(X)
        else:
            self.mean_ = (X * weights).sum(0) / weights.sum(0)

        # TODO: check for NaN and filter warnings
        Ws = weights.sum(0)
        XW = (X - self.mean_) * weights
        covar = np.dot(XW.T, XW) / np.dot(weights.T, weights)
        if self.xi != 0:
            covar *= np.outer(Ws, Ws) ** self.xi
        covar[np.isnan(covar)] = 0

        eigvals = (X.shape[1] - n_components, X.shape[1] - 1)
        evals, evecs = linalg.eigh(covar, eigvals=eigvals)
        self.components_ = evecs[:, ::-1].T
        self.explained_variance_ = evals[::-1]
        self.explained_variance_ratio_ = evals[::-1] / covar.trace()
        return self

    def transform(self, X, weights=None):
        if weights is None:
            weights = np.ones_like(X)
        Y = np.zeros((X.shape[0], self.components_.shape[0]))
        for i in range(X.shape[0]):
            W2 = weights[i] ** 2
            cWc = np.dot(self.components_ * W2, self.components_.T)
            cWX = np.dot(self.components_ * W2, X[i] - self.mean_)
            if self.regularize:
                cWc += np.diag(X.shape[0] / self.explained_variance_)
            Y[i] = np.linalg.solve(cWc, cWX)
        return Y

    def fit_transform(self, X, weights=None):
        return self.fit(X, weights).transform(X, weights)

    def inverse_transform(self, X):
        return self.mean_ + np.dot(X, self.components_)

    def reconstruct(self, X, weights=None):
        return self.inverse_transform(self.transform(X, weights))
