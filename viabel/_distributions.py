import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy import stats
from autograd.scipy import special
from autograd.numpy import linalg

import autograd.scipy.stats.multivariate_normal as mvn

# See: https://github.com/scipy/scipy/blob/master/scipy/stats/_multivariate.py
def multivariate_t_logpdf(x, m, S, df=np.inf):
    """calculate log pdf for each value

    Parameters
    ----------
    x : array_like, shape=(n_samples, n_features)

    m : array_like, shape=(n_features,)

    S : array_like, shape=(n_features, n_features)
        covariance  matrix
    df : int or float
        degrees of freedom
    """
    #m = np.asarray(m)
    d = m.shape[-1]
    if df == np.inf:
        return stats.multivariate_normal.logpdf(x, m, S)
    #psd = _PSD(S)
    s, u = linalg.eigh(S)
    eps = 1e-10
    s_pinv = np.array([0 if abs(x) <= eps else 1/x for x in s], dtype=float)
    U = np.multiply(u, np.sqrt(s_pinv))
    log_pdet = np.sum(np.log(s))

    log_pdf = special.gammaln(.5*(df + d)) - special.gammaln(.5*df) - .5*d * np.log(np.pi * df)
    log_pdf += -.5*log_pdet
    dev = x - m
    maha = np.sum(np.square(np.dot(dev, U)), axis=-1)
    log_pdf += -.5*(df + d) * np.log(1 + maha / df)
    return log_pdf

class MixtureMeanFieldGaussian():
    def __init__(self, dim, components):
        self.C = np.maximum(0, components)
        self.dim = np.maximum(1, dim)
        self.rs = npr.RandomState(0)
        means = [[0.] * dim] * self.C
        log_stds = [[1.] * dim] * self.C
        self.rho = (np.repeat(1, self.C)) / sum(np.repeat(1, self.C))
        self.params = [np.concatenate(pair) for pair in zip(means, log_stds)]

    def unpack_params(self, var_param):
        rho, mean, log_std = (var_param[0], var_param[1:(self.dim + 1)],
                              var_param[(self.dim + 2):])
        return rho, mean, log_std

    def add_component(self, var_param):
        self.C = self.C + 1
        rho, mean, log_std = self.unpack_params(var_param)
        self.params.append([mean, log_std])
        self.rho = self.rho * (1 - rho)
        self.rho = np.append(self.rho, rho)

    def sample(self, n_samples, seed):
        rs = self.rs if seed is None else npr.RandomState(seed)
        components = npr.choice(range(self.C), size = n_samples, p = self.rho)
        params = np.array([self.params[c] for c in components])
        means, log_stds = params[:, :self.dim], params[:, self.dim:]
        samples = rs.randn(n_samples, dim) * np.exp(log_stds) + means
        return samples

    def logpdf(self, x):
        params = np.array([self.params[c] for c in range(self.C)])
        means, log_stds = params[:, :self.dim], params[:, self.dim:]
        return np.sum([rho * mvn.logpdf(x, mean, np.diag(np.exp(2*log_std)))
                  for rho, mean, log_std in zip(self.rho, means, log_stds)])
