import numpy as np
from numpy.testing import assert_allclose


def orthonormalize(X, rows=True):
    """Orthonormalize X using QR-decomposition

    Parameters
    ----------
    X : array-like, [N, M]
        matrix to be orthonormalized
    rows : boolean (default=True)
        If True, orthonormalize rows of X. Otherwise orthonormalize columns.

    Returns
    -------
    Y : ndarray, [N, M]
        Orthonormalized version of X
    """
    orient = lambda X: X.T if rows else X
    Q, R = np.linalg.qr(orient(X))
    return orient(Q)


def random_orthonormal(N, M, rows=True, rseed=None):
    """Construct a random orthonormal matrix

    Parameters
    ----------
    N, M : integers
        The size of the matrix to construct.
    rows : boolean, default=True
        If True, return matrix with orthonormal rows.
        Otherwise return matrix with orthonormal columns.
    rseed : int or None
        Specify the random seed used in construction of the matrix.
    """
    if rows: assert N <= M
    else: assert N >= M
    rand = np.random.RandomState(rseed)
    return orthonormalize(rand.randn(N, M), rows=rows)


def solve_weighted(A, b, w):
    """solve Ax = b with weights w

    Parameters
    ----------
    A : array-like [N, M]
    b : array-like [N]
    w : array-like [N]

    Returns
    -------
    x : ndarray, [M]
    """
    A, b, w = map(np.asarray, (A, b, w))
    ATw2 = A.T * w ** 2
    return np.linalg.solve(np.dot(ATw2, A),
                           np.dot(ATw2, b))


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
        # for numerical stability, avoid doing this one-by-one
        d = data - np.dot(coeff[:, :i], eigvec[:i])
        c = coeff[:, i:i + 1]
        eigvec[i] = np.dot(c.T, w2 * d) / np.dot(c.T, w2 * c)
        # orthonormalize computed vectors: is this necessary at every step?
        eigvec[:i + 1] = orthonormalize(eigvec[:i + 1])
    return eigvec


def empca(data, weights, nvec, niter=25, rseed=None):
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
    eigvec = random_orthonormal(nvec, data.shape[1], rseed=rseed)
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
