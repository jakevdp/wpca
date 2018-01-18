import numpy as np
from scipy import linalg

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
from .utils import check_array_with_weights, weighted_mean


class WPCA(BaseEstimator, TransformerMixin):
    """Weighted Principal Component Analysis

    This is a direct implementation of weighted PCA based on the eigenvalue
    decomposition of the weighted covariance matrix following
    Delchambre (2014) [1]_.

    Parameters
    ----------
    n_components : int (optional)
        Number of components to keep. If not specified, all components are kept

    xi : float (optional)
        Degree of weight enhancement.

    regularization : float (optional)
        Control the strength of ridge regularization used to compute the
        transform.

    copy_data : boolean, optional, default True
        If True, X and weights will be copied; else, they may be overwritten.

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
    - sklearn.decomposition.PCA

    References
    ----------
    .. [1] Delchambre, L. MNRAS 2014 446 (2): 3545-3555 (2014)
           http://arxiv.org/abs/1412.4533
    """
    def __init__(self, n_components=None, xi=0, regularization=None,
                 copy_data=True):
        self.n_components = n_components
        self.xi = xi
        self.regularization = regularization
        self.copy_data = copy_data

    def _center_and_weight(self, X, weights, fit_mean=False):
        """Compute centered and weighted version of X and adjust weights.

        Input weights are inverse variance and adjusted weights are
        inverse sigmas. Will produce a RuntimeWarning if any input
        weights are negative.

        If fit_mean is True, then also save the mean to self.mean_
        """
        X, weights = check_array_with_weights(X, weights, dtype=float,
                                              copy=self.copy_data)

        # Convert from inverse variance to inverse sigmas.
        # See eqn. 7 of Delchambre 2015 and issue #2.
        weights = np.sqrt(weights)

        if fit_mean:
            self.mean_ = weighted_mean(X, weights, axis=0)

        # now let X <- (X - mean) * weights
        X -= self.mean_

        if weights is not None:
            X *= weights
        else:
            weights = np.ones_like(X)

        return X, weights

    def fit(self, X, y=None, weights=None):
        """Compute principal components for X

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.

        weights: array-like, shape (n_samples, n_features)
            Non-negative weights encoding the reliability of each measurement.
            Equivalent to the inverse variance when errors are Gaussian.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # let X <- (X - mean) * weights
        X, weights = self._center_and_weight(X, weights, fit_mean=True)
        self._fit_precentered(X, weights)
        return self

    def _fit_precentered(self, X, weights):
        """fit pre-centered data"""
        if self.n_components is None:
            n_components = X.shape[1]
        else:
            n_components = self.n_components

        # TODO: filter NaN warnings
        covar = np.dot(X.T, X)
        covar /= np.dot(weights.T, weights)
        covar[np.isnan(covar)] = 0

        # enhance weights if desired
        if self.xi != 0:
            Ws = weights.sum(0)
            covar *= np.outer(Ws, Ws) ** self.xi

        eigvals = (X.shape[1] - n_components, X.shape[1] - 1)
        evals, evecs = linalg.eigh(covar, eigvals=eigvals)
        self.components_ = evecs[:, ::-1].T
        self.explained_variance_ = evals[::-1]
        self.explained_variance_ratio_ = evals[::-1] / covar.trace()

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
            Equivalent to the inverse variance when errors are Gaussian.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        X, weights = self._center_and_weight(X, weights, fit_mean=False)
        return self._transform_precentered(X, weights)

    def _transform_precentered(self, X, weights):
        """transform pre-centered data"""
        # TODO: parallelize this?
        Y = np.zeros((X.shape[0], self.components_.shape[0]))
        for i in range(X.shape[0]):
            cW = self.components_ * weights[i]
            cWX = np.dot(cW, X[i])
            cWc = np.dot(cW, cW.T)
            if self.regularization is not None:
                cWc += np.diag(self.regularization / self.explained_variance_)
            Y[i] = np.linalg.solve(cWc, cWX)
        return Y

    def fit_transform(self, X, y=None, weights=None):
        """Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.

        weights: array-like, shape (n_samples, n_features)
            Non-negative weights encoding the reliability of each measurement.
            Equivalent to the inverse variance when errors are Gaussian.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        X, weights = self._center_and_weight(X, weights, fit_mean=True)
        self._fit_precentered(X, weights)
        return self._transform_precentered(X, weights)

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
        X = check_array(X)
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
            Equivalent to the inverse variance when errors are Gaussian.

        Returns
        -------
        X_reconstructed : ndarray, shape (n_samples, n_components)
            Reconstructed version of X
        """
        return self.inverse_transform(self.transform(X, weights=weights))

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
            Equivalent to the inverse variance when errors are Gaussian.

        Returns
        -------
        X_reconstructed : ndarray, shape (n_samples, n_components)
            Reconstructed version of X
        """
        return self.inverse_transform(self.fit_transform(X, weights=weights))
