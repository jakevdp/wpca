import numpy as np
from numpy.testing import assert_allclose


def norm_sign(X):
    i_max_abs = np.argmax(abs(X), 0)
    sgn = np.sign(X[i_max_abs, range(X.shape[1])])
    return X * sgn


def assert_columns_allclose_upto_sign(A, B, *args, **kwargs):
    assert_allclose(norm_sign(A), norm_sign(B), *args, **kwargs)
