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
        
    Notes
    -----
    dot(P, C) is the best approximation to the input X
    """
    nvar, nobs = X.shape
    U, s, VT = np.linalg.svd(X)
    return U[:, :ncomp], s[:ncomp, None] * VT[:ncomp], s[:ncomp] ** 2 / nobs


def wpca_delchambre(X, ncomp, W=None, xi=0):
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
    
    Returns
    -------
    P : ndarray [nvar, ncomp]
        The eigenvectors of the PCA
    C : ndarray [ncomp, nobs]
        The principal components
    sigma : ndarray [ncomp]
        The eigenvalues of the covariance
        
    Notes
    -----
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

    # compute the weighted covariance matrix
    Ws = W.sum(1, keepdims=True)
    XW = X * W
    covar = (Ws @ Ws.T) ** xi * (XW @ XW.T) / (W @ W.T)
    covar[np.isnan(covar)] = 0
    
    # diagonalize the weighted covariance
    evals, evecs = linalg.eigh(covar, eigvals=(nvar - ncomp, nvar - 1))
    sigma = evals[::-1]
    P = evecs[:, ::-1]

    # solve for the coefficients
    # TODO: can this be done more efficiently?
    C = np.zeros((ncomp, nobs))
    for i in range(nobs):
        W2 = np.diag(W[:, i] ** 2)
        C[:, i] = np.linalg.solve(P.T @ W2 @ P, P.T @ W2 @ X[:, i])

    return P, C, sigma


#----------------------------------------------------------------------
# Unit tests
from numpy.testing import assert_allclose


def norm_sign(X):
    # TODO: this may not work when column sum is zero
    sgn = np.sign(X.sum(0))
    sgn[sgn == 0] = 1
    return X * sgn


def assert_columns_allclose_upto_sign(A, B, *args, **kwargs):
    assert_allclose(norm_sign(A),
                    norm_sign(B), *args, **kwargs)


def test_wpca_delchambre_unweighted():
    rand = np.random.RandomState(0)
    X = rand.rand(5, 100)

    def check_results(ncomp):
        P1, C1, S1 = pca(X, ncomp)
        P2, C2, S2 = wpca_delchambre(X, ncomp)
        assert_allclose(S1, S2)
        assert_columns_allclose_upto_sign(P1, P2)
        assert_columns_allclose_upto_sign(C1.T, C2.T)

    for ncomp in range(1, 6):
        yield check_results, ncomp
