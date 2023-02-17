import numpy as np
from utils import logm


def frob(y: np.ndarray, yhat: np.ndarray) -> float:
    """Returns the Frobenius loss between y and yhat"""
    def _frob(a):
        return np.sqrt(np.trace(a.T @ a))
    return _frob(y - yhat)


def stein(y: np.ndarray, yhat: np.ndarray) -> float:
    """Returns the Stein loss between y and yhat"""
    def _stein(a):
        return np.trace(a) - np.log(np.linalg.det(a)) - a.shape[0]
    return _stein(y @ np.linalg.inv(yhat))


def vnd(y: np.ndarray, yhat: np.ndarray) -> float:
    """Returns the Von Neumann divergence loss between y and yhat"""
    def _vnd(a, b):
        return float(np.trace(a @ (logm(a) - logm(b)) - a + b))
    return _vnd(y, yhat)
