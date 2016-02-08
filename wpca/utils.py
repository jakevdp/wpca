import numpy as np
from sklearn.utils.validation import check_array


def check_array_with_weights(X, weights, **kwargs):
    """Utility to validate data and weights.

    This calls check_array on X and weights, making sure results match.
    """
    if weights is None:
        return check_array(X, **kwargs), weights

    # Always use copy=False for weights
    kwargs_weights = dict(kwargs)
    kwargs_weights.update(copy=False)
    weights = check_array(weights, **kwargs_weights)

    # Always use force_all_finite=False for X
    kwargs_X = dict(kwargs)
    kwargs_X.update(force_all_finite=False)
    X = check_array(X, **kwargs_X)

    # Make sure shapes match and missing data has weights=0
    if X.shape != weights.shape:
        raise ValueError("Shape of `X` and `weights` should match")

    Wzero = (weights == 0)
    X[Wzero] = 0

    if not np.all(np.isfinite(X)):
        raise ValueError("Input contains NaN or infinity without "
                         "a corresponding zero in `weights`.")
    return X, weights


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


def random_orthonormal(N, M, rows=True, random_state=None):
    """Construct a random orthonormal matrix

    Parameters
    ----------
    N, M : integers
        The size of the matrix to construct.
    rows : boolean, default=True
        If True, return matrix with orthonormal rows.
        Otherwise return matrix with orthonormal columns.
    random_state : int or None
        Specify the random state used in construction of the matrix.
    """
    assert N <= M if rows else N >= M
    rand = np.random.RandomState(random_state)
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
