import numpy as np
from numpy.testing import assert_allclose

def orthonormalize(X, rows=True):
    if rows: X = X.T
    Q, R = np.linalg.qr(X)
    if rows: Q = Q.T
    return Q


def random_orthonormal(n_samples, n_features, rseed=None):
    assert(n_samples <= n_features)
    rand = np.random.RandomState(rseed)
    X = rand.randn(n_samples, n_features)
    return orthonormalize(X)


def solve_weighted(A, b, w):
    """solve Ax = b with weights w"""
    return np.linalg.solve(np.dot(A.T * w ** 2, A),
                           np.dot(A.T * w ** 2, b))


def _Estep(eigvec, data, weights, coeff):
    for i in range(data.shape[0]):
        coeff[i] = solve_weighted(eigvec.T, data[i], weights[i])
    return coeff


def _Mstep(eigvec, data, weights, coeff):
    data = data.copy()
    for i in range(eigvec.shape[0]):
        c = coeff[:, i]
        for j in range(data.shape[1]):
            w = weights[:, j]
            x = data[:, j]
            eigvec[i, j] = np.dot(x, w ** 2 * c) / np.dot(c, w ** 2 * c)

        # remove this vector from the data
        data -= np.outer(coeff[:, i], eigvec[i])

    return orthonormalize(eigvec)


def empca(data, weights, nvec, niter=25, rseed=None):
    assert data.shape == weights.shape
    assert nvec <= data.shape[1]
    eigvec = random_orthonormal(nvec, data.shape[1], rseed=rseed)
    coeff = np.zeros((data.shape[0], nvec))

    for k in range(niter):
        coeff = _Estep(eigvec, data, weights, coeff)
        eigvec = _Mstep(eigvec, data, weights, coeff)
    coeff = _Estep(eigvec, data, weights, coeff)

    return eigvec, coeff


def pca(data, nvec):
    U, s, VT = np.linalg.svd(data)
    return VT[:, :nvec], U[:, :nvec] * s[:nvec]
