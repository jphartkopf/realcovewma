import numpy as np
import scipy.optimize as sopt
import scipy.special as ssp


class Model:
    """Model framework class. Only for inheritance"""
    def __init__(self):
        pass

    def fit(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass


class EWMA(Model):
    """
    Exponentially weighted moving average for prediction of realized covariances.
    Smoothing parameter is set by calling EWMA.fit()
    """
    def __init__(self):
        super().__init__()
        self.lam = None  # fit() model to set param

    def fit(self, **kwargs):
        self.lam = np.clip(kwargs["lam"], 0., 1.)

    def predict(self, lagged_y, lagged_yhat):
        if self.lam is None:
            self.fit(lam=0.9)
            print("No smoothing parameter specified. Set to default value (= 0.9)."
                  " Please use EWMA.fit() method to specify different parameter.")
        return self.lam * lagged_yhat + (1 - self.lam) * lagged_y


class RandomWalk(EWMA):
    """
    Random walk model for prediction of realized covariances.
    Smoothing parameter is tied to lam = 0.0
    """
    def __init__(self):
        super().__init__()
        self.lam = 0.0  # not fitting necessary

    def fit(self):
        """Prevent altering of the smoothing parameter"""
        self.lam = 0.0


class UhligExtension(EWMA):
    """
    Uhlig extension model of Windle and Carvalho (2014)

    References:
        [1] Windle, J. and Carvalho, C. (2014). "A Tractable State-Space Model for Symmetric
            Positive-Definite Matrices." Bayesian Analysis, 9(4): 759-792.
        [2] Hartkopf, J. P. (2023). "Composite forecasting of vast-dimensional realized
            covariance matrices using factor state-space models." Empirical Economics 64(1),
            393-436.
    """
    def __init__(self):
        super().__init__()
        self.k = None  # fit() model to set param
        self.n = None  # fit() model to set param

    def fit(self, y, k0, n0):
        m, _, t = y.shape
        x0 = np.array([k0, n0])
        bnds = ((m, None), (m, None))
        method = "trust-constr"
        self._fit(self._log_likelihood, x0, args=(y), method=method, bounds=bnds)
        self.lam = self._restrict_lam(self.k, self.n, m)

    def _fit(
            self,
            fun,
            x0,
            args=(),
            method="trust-constr",
            bounds=None,
            tol=None
    ):
        res = sopt.minimize(fun, x0, args, method=method, bounds=bounds, tol=tol)
        k, n = res.x[0], res.x[1]
        self.k = k
        self.n = n

    def _log_likelihood(self, x, y):
        m, _, t = y.shape
        k, n = x
        lam = self._restrict_lam(k, n, m)
        s = self._initialize_s(y, lam)
        llh = 0.
        for t_ in np.arange(t):
            y_ = y[:, :, t_]
            llh -= self._likelihood_contribution(k, n, m, y_, s)
            s = lam * s + y_
        return llh

    def _likelihood_contribution(self, k, n, m, y, s):
        lam = self._restrict_lam(k, n, m)
        contribution = ssp.multigammaln(0.5*(k+n), m)
        contribution -= ssp.multigammaln(0.5*n, m)
        contribution -= ssp.multigammaln(0.5*k, m)
        contribution += 0.5*(k-m-1)*np.log(np.linalg.det(y))
        contribution += 0.5*n*np.log(np.linalg.det(lam*s))
        contribution -= 0.5*(n+k)*np.log(np.linalg.det(lam*s+y))
        return contribution

    @staticmethod
    def _initialize_s(y, *args, **kwargs):
        s = np.eye(y.shape[0])
        return s

    @staticmethod
    def _restrict_lam(k, n, m):
        lam = 1 / (1 + k / (n - m - 1))
        return np.clip(lam, 0., 1.)
