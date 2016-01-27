import numpy as np
from numpy.testing import assert_allclose


def orthonormalize(X, rows=True):
    orient = lambda X: X.T if rows else X
    Q, R = np.linalg.qr(orient(X))
    return orient(Q)


def random_orthonormal(n_samples, n_features, rseed=None):
    assert(n_samples <= n_features)
    rand = np.random.RandomState(rseed)
    X = rand.randn(n_samples, n_features)
    return orthonormalize(X)


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
    return np.linalg.solve(np.dot(A.T * w ** 2, A),
                           np.dot(A.T * w ** 2, b))


def _Estep(eigvec, data, weights, coeff):
    # Update coeff
    for i in range(data.shape[0]):
        coeff[i] = solve_weighted(eigvec.T, data[i], weights[i])


def _Mstep(eigvec, data, weights, coeff):
    # Update eigvec
    w2 = weights ** 2
    for i in range(eigvec.shape[0]):
        d = data - np.dot(coeff[:, :i], eigvec[:i])
        c = coeff[:, i:i + 1]
        eigvec[i] = np.dot(c.T, w2 * d) / np.dot(c.T, w2 * c)
        eigvec[:i + 1] = orthonormalize(eigvec[: i + 1])


def empca(data, weights, nvec, niter=25, rseed=None):
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
    U, s, VT = np.linalg.svd(data)
    return VT[:, :nvec], U[:, :nvec] * s[:nvec]
