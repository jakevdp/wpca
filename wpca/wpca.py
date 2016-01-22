"""
Weighted Principal Component Analysis
"""
import numpy as np
from scipy import linalg


def pca(X, ncomp):
    """Standard Principal Component Analysis

    This is standard PCA, implemented using a singular value decomposition

    Parameters
    ----------
    X : array_like [nvar, nobs]
        input data; each column represents an observation.
        assumed to be pre-centered.
    ncomp : integer
        number of components to return

    Returns
    -------
    P : ndarray [nvar, ncomp]
        The eigenvectors of the PCA
    C : ndarray [ncomp, nobs]
        The principal components
    sigma : ndarray [ncomp]
        The eigenvalues of the covariance
    var_tot : float
        Total variance: i.e. sum of all eigenvalues

    Notes
    -----
    dot(P, C) is the best approximation to the input X.
    """
    nvar, nobs = X.shape
    U, s, VT = np.linalg.svd(X, full_matrices=False)
    sigma = s ** 2 / nobs
    var_tot = sigma.sum()
    return U[:, :ncomp], s[:ncomp, None] * VT[:ncomp], sigma[:ncomp], var_tot


def wpca_delchambre(X, ncomp, W=None, xi=0, compute_transform=True,
                    regularize=False):
    """Weighted Principal Component Analysis

    This is a direct implementation of weighted PCA based on the eigenvalue
    decomposition of the weighted covariance matrix following
    Delchambre (2014) [1]_.

    Parameters
    ----------
    X : array_like [nvar, nobs]
        input data; each column represents an observation
        assumed to be pre-centered
    ncomp : integer
        number of components to return
    W : array_like [nvar, nobs] (default = None)
        weights associated with the data X
    xi : float (defaut = 0)
        degree of enhancement for weights. Default 0 (no effect)
    compute_transform : bool (default=True)
        if True, compute and return transform matrix C
    regularize : bool
        [TODO]

    Returns
    -------
    P : ndarray [nvar, ncomp]
        The eigenvectors of the PCA
    C : ndarray [ncomp, nobs]
        The principal components (or None if compute_transform is False)
    sigma : ndarray [ncomp]
        The eigenvalues of the covariance
    var_tot : float
        Total variance: i.e. sum of all eigenvalues

    Notes
    -----
    dot(P, C) is the best approximation to the input X.

    Although Delchambre (2014) suggests an iterative power method, this
    iteration is not essential to the algorithm. Fundamentally the algorithm
    is simply finding the standard eigenvectors of the weighted covariance
    matrix, and so we use typical scipy tools here.

    References
    ----------
    .. [1] Delchambre, L. MNRAS 2014 446 (2): 3545-3555 (2014)
           http://arxiv.org/abs/1412.4533
    """
    nvar, nobs = X.shape

    if W is None:
        W = np.ones((nvar, nobs))

    if W.shape != X.shape:
        raise ValueError("shape of W and X must match")

    # compute the weighted covariance matrix; shape is [nvar, nvar]
    Ws = W.sum(1, keepdims=True)
    XW = X * W
    covar = (Ws @ Ws.T) ** xi * (XW @ XW.T) / (W @ W.T)
    covar[np.isnan(covar)] = 0

    # diagonalize the weighted covariance
    evals, evecs = linalg.eigh(covar, eigvals=(nvar - ncomp, nvar - 1))
    sigma = evals[::-1]
    P = evecs[:, ::-1]
    C = None

    if compute_transform:
        # solve for the coefficients
        # TODO: can this be done more efficiently?
        C = np.zeros((ncomp, nobs))
        for i in range(nobs):
            W2 = np.diag(W[:, i] ** 2)
            if regularize:
                PWP = P.T @ W2 @ P + np.diag(X.shape[0] / sigma)
            else:
                PWP = P.T @ W2 @ P
            C[:, i] = np.linalg.solve(PWP, P.T @ W2 @ X[:, i])

    return P, C, sigma, covar.trace()
