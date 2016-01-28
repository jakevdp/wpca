import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .utils import orthonormalize, random_orthonormal, solve_weighted


class EMPCA(BaseEstimator, TransformerMixin):
    """Expectation-Maximization PCA"""
    def __init__(self, n_components=None, max_iter=100, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.random_state = random_state

    def _Estep_weighted(self, eigvec, data, weights, coeff):
        """E-step for weighted EMPCA: update coeff"""
        for i in range(data.shape[0]):
            coeff[i] = solve_weighted(eigvec.T, data[i], weights[i])
        return coeff

    def _Estep_unweighted(self, eigvec, data, coeff):
        """E-step for unweighted EMPCA: update coeff"""
        coeff[:] = np.linalg.solve(eigvec.T, data.T).T
        return coeff

    def _Mstep_weighted(self, eigvec, data, weights, coeff):
        """M-step for weighted EMPCA: update eigvec"""
        w2 = weights ** 2
        for i in range(eigvec.shape[0]):
            # remove contribution of previous eigenvectors from data
            d = data - np.dot(coeff[:, :i], eigvec[:i])
            c = coeff[:, i:i + 1]
            eigvec[i] = np.dot(c.T, w2 * d) / np.dot(c.T, w2 * c)
            # orthonormalize computed vectors: in theory not necessary,
            # but numerically it's a good idea
            # TODO: perhaps do this more efficiently?
            eigvec[:i + 1] = orthonormalize(eigvec[:i + 1])
        return eigvec

    def _Mstep_unweighted(self, eigvec, data, coeff):
        """M-step for unweighted EMPCA: update eigvec"""
        for i in range(eigvec.shape[0]):
            # remove contribution of previous eigenvectors from data
            d = data - np.dot(coeff[:, :i], eigvec[:i])
            c = coeff[:, i:i + 1]
            eigvec[i] = np.dot(c.T, d) / np.dot(c.T, c)
            # orthonormalize computed vectors: in theory not necessary,
            # but numerically it's a good idea
            # TODO: perhaps do this more efficiently?
            eigvec[:i + 1] = orthonormalize(eigvec[:i + 1])
        return eigvec

    def fit_transform(self, X, weights=None):
        if weights is None:
            self.mean_ = X.mean(0)
            weights = np.ones_like(X)
        else:
            XW = X * weights
            # handle NaN values
            XW[weights == 0] = 0
            self.mean_ = XW.sum(0) / weights.sum(0)

        X_c = X - self.mean_

        eigvec = random_orthonormal(self.n_components, X.shape[1],
                                    random_state=self.random_state)
        coeff = np.zeros((X.shape[0], self.n_components))

        # TODO: add a convergence check
        if weights is None:
            for k in range(self.max_iter):
                self._Estep_unweighted(eigvec, X_c, coeff)
                self._Mstep_unweighted(eigvec, X_c, coeff)
            self._Estep_unweighted(eigvec, X_c, coeff)
        else:
            for k in range(self.max_iter):
                self._Estep_weighted(eigvec, X_c, weights, coeff)
                self._Mstep_weighted(eigvec, X_c, weights, coeff)
            self._Estep_weighted(eigvec, X_c, weights, coeff)

        self.components_ = eigvec
        # TODO: need to compute these:
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        return coeff

    def fit(self, X, weights=None):
        self.fit_transform(X, weights=weights)
        return self

    def transform(self, X, weights=None):
        coeff = np.zeros((X.shape[0], self.n_components))
        if weights is None:
            return self._Estep_unweighted(self.components_, X, coeff)
        else:
            return self._Estep_weighted(self.components_, X, weights, coeff)


def _Estep(eigvec, data, weights, coeff):
    """E-step for Expectation Maximization PCA"""
    # Update coeff
    for i in range(data.shape[0]):
        coeff[i] = solve_weighted(eigvec.T, data[i], weights[i])
    return coeff


def _Mstep(eigvec, data, weights, coeff):
    """M-step for Expectation Maximization PCA"""
    # Update eigvec
    w2 = weights ** 2
    for i in range(eigvec.shape[0]):
        # remove contribution of previous eigenvectors from data
        d = data - np.dot(coeff[:, :i], eigvec[:i])
        c = coeff[:, i:i + 1]
        eigvec[i] = np.dot(c.T, w2 * d) / np.dot(c.T, w2 * c)
        # orthonormalize computed vectors: in theory not necessary,
        # but numerically it's a good idea
        # TODO: perhaps do this more efficiently?
        eigvec[:i + 1] = orthonormalize(eigvec[:i + 1])
    return eigvec


def empca(data, weights, nvec, niter=25, random_state=None):
    """Expectation-Maximization weighted PCA

    This computes an iterative weighted PCA of the data
    using the method of Bailey (2012) [1]_.

    Parameters
    ----------

    Returns
    -------

    References
    ----------
    .. [1] Bailey, S. PASP (2014)
           http://arxiv.org/abs/1208.4122
    """
    assert data.shape == weights.shape
    assert nvec <= data.shape[1]
    eigvec = random_orthonormal(nvec, data.shape[1],
                                random_state=random_state)
    coeff = np.zeros((data.shape[0], nvec))

    for k in range(niter):
        _Estep(eigvec, data, weights, coeff)
        _Mstep(eigvec, data, weights, coeff)
    _Estep(eigvec, data, weights, coeff)

    return eigvec, coeff


def pca(data, nvec):
    """Standard PCA using Singular Value Decomposition"""
    U, s, VT = np.linalg.svd(data)
    return VT[:, :nvec], U[:, :nvec] * s[:nvec]
