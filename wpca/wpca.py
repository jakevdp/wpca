import numpy as np
from scipy import linalg

from sklearn.base import BaseEstimator, TransformerMixin


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
    def __init__(self, n_components=None, xi=0, regularization=None):
        self.n_components = n_components
        self.xi = xi
        self.regularization = regularization

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
        if self.n_components is None:
            n_components = min(X.shape)
        else:
            n_components = self.n_components

        if weights is None:
            self.mean_ = X.mean(0)
            weights = np.ones_like(X)
        else:
            XW = X * weights
            # handle NaN values
            XW[weights == 0] = 0
            self.mean_ = XW.sum(0) / weights.sum(0)

        # TODO: check for NaN and filter warnings
        Ws = weights.sum(0)
        XW = (X - self.mean_) * weights

        # Handle NaNs in XW
        XW[weights == 0] = 0

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
        if weights is None:
            weights = np.ones_like(X)
        Y = np.zeros((X.shape[0], self.components_.shape[0]))
        for i in range(X.shape[0]):
            cW = self.components_ * weights[i]
            WX = (X[i] - self.mean_) * weights[i]
            # handle NaN values in X
            WX[weights[i] == 0] = 0

            cWc = np.dot(cW, cW.T)
            cWX = np.dot(cW, WX)
            if self.regularization is not None:
                cWc += np.diag(self.regularization / self.explained_variance_)
            Y[i] = np.linalg.solve(cWc, cWX)
        return Y

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
        return self.fit(X, weights).transform(X, weights)

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
        X_reconstructed : array-like, shape (n_samples, n_components)
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
        X_reconstructed : array-like, shape (n_samples, n_components)
            Reconstructed version of X
        """
        return self.inverse_transform(self.fit_transform(X, weights))
