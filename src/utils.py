import numpy as np


def zeros_matrix(*args, **kwargs):
    return np.matrix(np.zeros(*args, **kwargs))


def vec(matrix: np.ndarray | np.matrix) -> np.ndarray | np.matrix:
    return matrix.flatten().T


def vec2mat(vector):
    n = int(np.sqrt(len(vector)))
    return np.reshape(vector, (n, n))


def logm(a):
    """Calculates the matrix logarithm of square matrix a"""
    _, v = np.linalg.eig(a)
    v_inv = np.linalg.inv(v)
    a_prime = v_inv @ a @ v
    diag_a_prime = np.diag(a_prime)
    a_ = a_prime - np.diag(diag_a_prime)
    a_ += np.diag(np.log(diag_a_prime))
    return v @ a_ @ v_inv


def duplication_matrix(n: int) -> np.ndarray:
    """
    Returns a duplication matrix D of order n s.th.
        D * vech(A) = vec(A)
    where A is a (n x n) symmetric matrix, vec() its vectorization and
    vech() its lower triangular vectorization.

    :param n:   int         (Dimension of symmetric matrix A)
    :return:    np.matrix   (Duplication matrix of size n^2 x n*(n+1)/2)
    """
    n_sq = pow(n, 2)
    n_bar = n * (n+1) // 2
    D = zeros_matrix(shape=(n_sq, n_bar), dtype=int)
    for j_ in np.arange(n):
        for i_ in np.arange(j_, n):
            u = zeros_matrix(shape=(n_bar, 1), dtype=int)
            u[j_ * n + i_ - ((j_ + 1) * j_) // 2] = 1
            t = zeros_matrix(shape=(n, n), dtype=int)
            t[i_, j_] = 1
            t[j_, i_] = 1
            D += vec(t) * u.T
    return np.array(D)
