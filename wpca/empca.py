import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .utils import orthonormalize, random_orthonormal, solve_weighted


class EMPCA(BaseEstimator, TransformerMixin):
    """Expectation-Maximization PCA

    This is an iterative implementation of weighted PCA based on an
    Expectation-Maximization approach, following Bailey (2012) [1]_.

    Parameters
    ----------
    n_components : int (optional)
        Number of components to keep. If not specified, all components are kept

    max_iter : int (default=100)
        Maximum number of Expectation-Maximization iterations

    random_state : int or None
        Seed for the random initialization of eigenvectors

    Attributes
    ----------
    components_ : array, [n_components, n_features]
        Principal axes in feature space, representing the directions of
        maximum variance in the data.

    explained_variance_ : array, [n_components]
        The amount of variance explained by each of the selected components.

    explained_variance_ratio_ : array, [n_components]
        Percentage of variance explained by each of the selected components.

    mean_ : array, [n_features]
        Per-feature empirical mean, estimated from the training set.

    See Also
    --------
    - PCA
    - WPCA
    - sklearn.decomposition.PCA

    References
    ----------
    .. [1] Bailey, S. PASP 124:919 (2012)
           http://arxiv.org/abs/1208.4122
    """
    def __init__(self, n_components=None, max_iter=100, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.random_state = random_state

    def _Estep(self, data, weights, eigvec):
        """E-step: solve for coeff given eigvec"""
        if weights is None:
            return np.dot(data, eigvec.T)
        else:
            return np.array([solve_weighted(eigvec.T, data[i], weights[i])
                             for i in range(data.shape[0])])

    def _Mstep(self, data, weights, eigvec, coeff):
        """M-step: solve for eigvec given coeff"""
        w2 = 1 if weights is None else weights ** 2

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

    def fit_transform(self, X, weights=None):
        """Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.

        weights: array-like, shape (n_samples, n_features)
            Non-negative weights encoding the reliability of each measurement.
            Equivalent to the inverse of the Gaussian errorbar.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        if self.n_components is None:
            n_components = min(X.shape)
        else:
            n_components = self.n_components

        if weights is None:
            self.mean_ = X.mean(0)
            X_c = X - self.mean_
        else:
            assert X.shape == weights.shape
            XW = X * weights
            XW[weights == 0] = 0
            self.mean_ = XW.sum(0) / weights.sum(0)
            X_c = X - self.mean_
            X_c[weights == 0] = 0

        eigvec = random_orthonormal(n_components, X.shape[1],
                                    random_state=self.random_state)

        # TODO: add a convergence check
        for k in range(self.max_iter):
            coeff = self._Estep(X_c, weights, eigvec)
            eigvec = self._Mstep(X_c, weights, eigvec, coeff)
        coeff = self._Estep(X_c, weights, eigvec)

        self.components_ = eigvec
        self.explained_variance_ = (coeff ** 2).sum(0) / X.shape[0]

        # TODO: correctly handle weighted variance here?
        self.explained_variance_ratio_ = (self.explained_variance_
                                          / X_c.var(0).sum())
        return coeff

    def fit(self, X, weights=None):
        """Compute principal components for X

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.

        weights: array-like, shape (n_samples, n_features)
            Non-negative weights encoding the reliability of each measurement.
            Equivalent to the inverse of the Gaussian errorbar.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.fit_transform(X, weights=weights)
        return self

    def transform(self, X, weights=None):
        """Apply dimensionality reduction on X.

        X is projected on the first principal components previous extracted
        from a training set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.

        weights: array-like, shape (n_samples, n_features)
            Non-negative weights encoding the reliability of each measurement.
            Equivalent to the inverse of the Gaussian errorbar.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        X_c = X - self.mean_
        if weights is not None:
            assert X.shape == weights.shape
            X_c[weights == 0] = 0
        return self._Estep(X_c, weights, self.components_)

    def inverse_transform(self, X):
        """Transform data back to its original space.

        Returns an array X_original whose transform would be X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            Data in transformed representation.

        Returns
        -------
        X_original : array-like, shape (n_samples, n_features)
        """
        return self.mean_ + np.dot(X, self.components_)

    def reconstruct(self, X, weights=None):
        """Reconstruct the data using the PCA model

        This is equivalent to calling transform followed by inverse_transform.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            Data in transformed representation.

        weights: array-like, shape (n_samples, n_features)
            Non-negative weights encoding the reliability of each measurement.
            Equivalent to the inverse of the Gaussian errorbar.

        Returns
        -------
        X_reconstructed : ndarray, shape (n_samples, n_components)
            Reconstructed version of X
        """
        return self.inverse_transform(self.transform(X, weights))

    def fit_reconstruct(self, X, weights=None):
        """Fit the model and reconstruct the data using the PCA model

        This is equivalent to calling fit_transform()
        followed by inverse_transform().

        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            Data in transformed representation.

        weights: array-like, shape (n_samples, n_features)
            Non-negative weights encoding the reliability of each measurement.
            Equivalent to the inverse of the Gaussian errorbar.

        Returns
        -------
        X_reconstructed : ndarray, shape (n_samples, n_components)
            Reconstructed version of X
        """
        return self.inverse_transform(self.fit_transform(X, weights))
