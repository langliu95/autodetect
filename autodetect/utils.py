"""Utils for autograd-test.

Author: Lang Liu
Date: 06/10/2019
"""

from __future__ import absolute_import, division, print_function

import math

import numpy as np
from scipy.special import chdtri
from scipy.stats import chi2
import torch
import torch.nn as nn


##########################################################################
# functions for models
##########################################################################

def loglike(outs, targets):
    """Log-likelihood of Gaussian."""
    loss_fn = nn.MSELoss(reduction='sum')
    return -loss_fn(outs, targets) / 2


def loglike_linear(pars, obs):
    """Log-likelihood of linear models."""
    inputs = obs[0]
    inputs = torch.cat([inputs, torch.ones(len(inputs), 1)], 1)
    outs = inputs @ pars[0]
    return loglike(outs, obs[1])


def loglike_hmm(pars, obs, init='uniform'):
    """Log-likelihood of hidden Markov models."""
    trans = pars[0]
    last_col = 1 - torch.sum(trans, 1, keepdim=True)
    trans = torch.cat((trans, last_col), 1)
    emis = pars[1]
    obs = obs[0]
    N = len(trans)
    
    def loglike_emission(y):
        """Gaussian emission."""
        return -((y - emis[:, 0]) / emis[:, 1])**2 / 2 -\
            torch.tensor(np.log(2 * math.pi)) / 2 - torch.log(emis[:, 1])
    
    if init == 'uniform':
        init = torch.ones(N) / N
    if init == 'random':
        init = torch.rand(N)
        init = init / torch.sum(init)
    
    loglike = torch.zeros(len(obs))
    g = torch.exp(loglike_emission(obs[0]))
    c = torch.sum(g * init)
    phi = init * g / c
    loglike[0] = torch.log(c)
    
    for i, y in enumerate(obs[1:]):
        g = torch.exp(loglike_emission(y))
        alpha = phi @ trans * g
        c = torch.sum(alpha)
        phi = alpha / c
        loglike[i+1] = torch.log(c)
    
    return torch.sum(loglike)


##########################################################################
# functions for computing thresholds
##########################################################################

def _log_comb(a, b, part):
    """Compute log combinatorics choosing a from b.

    Parameters
    ----------
    part: array-like, shape (d,)
        Partial sums of :math:`\\log(d!)`, where :math:`a, b <= d`.
    """
    if (a <= b) & (a >= 0):
        return part[b] - part[a] - part[b - a]
    return -math.inf


def _log_card(p, d):
    """Compute log cardinality of the feasible space.

    Parameters
    ----------
    p: int
        Number of changed components.
    d: int
        Dimension of parameters of interest.

    Return:
        Log cardinality of the feasible space.
    """
    log_part_sum = np.concatenate((np.array([0]),
                                   np.cumsum(np.log(np.arange(d) + 1))))
    return _log_comb(p, d, log_part_sum)


def _theo_crit(p, d, n, alpha):
    """Compute theoretical critical value H_{p,n}.

    Parameters
    ----------
    p: int
        Number of changed components.
    d: int
        Dimension of parameters of interest (in which change is detected).
    """

    log_prob = math.log(alpha) - math.log(n) - _log_card(p, d) - 2 * math.log(p + 1)
    return (chdtri(p, math.exp(log_prob)) - p) / math.sqrt(2 * p)


def _lin_threshold(d, n, alpha):
    """Compute critical value of chi-square distributions."""
    return (chi2.ppf(1 - alpha / n, d) - d) / math.sqrt(2 * d)


def _scan_thresholds(d, n, alpha, prange):
    """Compute thresholds for scan statistics."""
    thresh = np.zeros(len(prange))
    for j, p in enumerate(prange):
        thresh[j] = _theo_crit(p, d, n, alpha)
    return thresh


def compute_thresholds(d, n, alpha, prange=None, stat_type='all'):
    thresh = [False] * 4
    if stat_type in ['linear', 'all']:
        thresh[0] = _lin_threshold(d, n, alpha[0])
    if stat_type in ['scan', 'all']:
        thresh[1] = _scan_thresholds(d, n, alpha[1], prange)
    if stat_type in ['autograd', 'all']:
        thresh[2] = _lin_threshold(d, n, alpha[2])
        thresh[3] = _scan_thresholds(d, n, alpha[3], prange)
    return thresh


def sample_square_Bessel(df, path_size, T=1):
    """Generate a sample path from square Bessel process.

    Parameters
    ----------
    df: int
        Degree of freedom.
    path_size: int
        Number of observations on the sample path.
    T: float, optional
        End time of the square Bessel process. Default is 1.
    """
    h = T / path_size
    Z = np.random.randn(path_size)
    path = np.zeros(path_size)
    for i in range(path_size-1):
        path[i+1] = path[i] + df * h + 2 * np.sqrt(path[i] * h) * Z[i+1] + \
            h * (Z[i+1]**2 - 1)
    return path


def sample_max_square_Bessel(df, path_size, T=1):
    """Generate a sample from the maximum of square Bessel process.

    Parameters
    ----------
    df: int
        Degree of freedom.
    path_size: int
        Number of observations on the sample path.
    T: float, optional
        End time of the square Bessel process. Default is 1.
    """
    path = sample_square_Bessel(df, path_size, T)[1:]  # discard 0
    h = T / path_size
    U = np.random.uniform(size=path_size-2)
    log_M = np.zeros(path_size-2)
    for i in range(path_size-2):
        log_M[i] = (np.log(path[i] * path[i+1]) + np.sqrt(np.log(path[i+1] / \
            path[i])**2 - 8 * h * np.log(U[i]) / path[i])) / 2
    return np.exp(np.max(log_M))


def quantile_max_square_Bessel(q, df, path_size, size, T=1):
    """Generate quantiles of the maximum of square Bessel process.

    Parameters
    ----------
    q: float
        Percentile.
    df: int
        Degree of freedom.
    path_size: int
        Number of observations on the sample path.
    size: int
        Number of observations used to estimate the quantile.
    T: float, optional
        End time of the square Bessel process. Default is 1.
    """
    sample = np.zeros(size)
    quant = np.zeros(int(size / 2))
    for i in range(size):
        sample[i] = sample_max_square_Bessel(df, path_size, T)
        if i >= int(size / 2):
            quant[i - int(size/2)] = np.quantile(sample[:i], q)
    return quant


##########################################################################
# functions for computing statistics
##########################################################################

def _get_batch(obs, lo, hi):
    """Get a batch of observations."""
    return [ob[lo:hi] for ob in obs]

def _compute_inv(ident, info, const=0.0):
    """Compute inverse of observed information."""
    while True:
        try:
            Iinv, _ = torch.solve(ident, info + const * ident)
            break
        except RuntimeError:
            print("Observed information matrix is ill conditioned.")
            print("Adding a scalar matrix for an approximation.")
            print("The result might be less informative unless the sample size is large enough.")
            const += 0.1
    return Iinv


def _max_score(p, order, score, Ischur):
    """Compute maximal score statistic over index sets with fixed cardinality.

    Parameters
    ----------
    p: int
        Cardinality of the index set.
    order: array-like, shape (d,)
        Order of ``(score ** 2) / np.diag(info)``.
    score: array-like, shape (d,)
        Score function w.r.t parameters with a change at a given location.
    Ischur: array-like, shape (d, d)
        Schur complement of the information matrix.

    Returns
    -------
    stat: torch.Tensor
        Maximal score statistic (unnormalized). It is ``False`` if the submatrix of ``Ischur`` is ill-conditioned.
    """
    index = order[-p:]
    sub_score = score[index]
    sub_Ischur = Ischur[np.ix_(index, index)]
    try:
        if type(score).__module__ == 'numpy':
            inv = np.linalg.solve(sub_Ischur, sub_score)
        else:
            inv, _ = torch.solve(sub_score.view(-1, 1), sub_Ischur)
        stat = sub_score @ inv
    except (RuntimeError, np.linalg.LinAlgError) as _:
        # print("Conditional information matrix is not invertible.")
        # print("Adding a scalar matrix for an approximation.")
        # inv, _ = torch.solve(sub_score.view(-1, 1), sub_Ischur + 0.1*torch.eye(p))
        stat = False  # if sub_Ischur is ill-conditioned, set the stat to False
    return stat


def _compute_normalized_stat(stat, df, thresh):
    """Compute normalized statistic.

    Parameters
    ----------
    stat : torch.Tensor
        Unnormalized score statistic.
    df : int
        Degree of freedom
    thresh : float or bool
        Threshold for a given significance level. Returns ``0.0`` if ``False``.
    """
    if stat is False:
        new_stat = 1.0
    elif not thresh:
        new_stat = 0.0
    else:
        new_stat = (stat - df) / math.sqrt(2 * df) / thresh
    return new_stat


def _compute_stats(prange, idx, score, info, Iinv, thresh,
                   stat_type='all', normalization='schur', finfo=None):
    """Compute statistics for change point detection.

    Parameters
    ----------
    prange : array-like
        Feasible set of the number of changed components. Set to `idx` for linear statistic.
    idx : array-like
        Indices of parameters of interest.
    score : torch.Tensor, shape (dim,)
        Conditional score function.
    info : torch.Tensor, shape (dim, dim)
        Conditional observed information.
    finfo : torch.Tensor, shape (dim, dim)
        The full observed information or the inverse of it (``'schur'``).
    thresh : list
        Thresholds for the tests, ``[lin, scan, Autograd_lin, Autograd_scan]``.
    stat_type : str, optional
        Type of the test statistic. Default is 'all'.
    """

    dim = len(idx)
    if type(score).__module__ == 'numpy':
        stat = np.zeros(3)  # [linear, scan, score]
        new_stat = np.zeros(3)
    else:
        stat = torch.zeros(3)  # [linear, scan, score]
        new_stat = torch.zeros(3)
    index = [idx, np.arange(max(prange)), idx]

    # linear statistics
    if normalization == 'schur':
        Ischur = info[np.ix_(idx, idx)] - info[idx, :] @ Iinv @ info[:, idx]
        new_score = _max_score(0, range(dim), score[idx], Ischur)
    else:
        new_score = _max_score(0, range(dim), score[idx], info[np.ix_(idx, idx)])
        new_score += _max_score(
            0, range(dim), score[idx], (finfo - info)[np.ix_(idx, idx)])
    new_stat[0] = _compute_normalized_stat(new_score, dim, thresh[0])
    new_stat[2] = _compute_normalized_stat(new_score, dim, thresh[2])
    if new_stat[0] > stat[0]: stat[0] = new_stat[0]
    if new_stat[2] > stat[2]: stat[2] = new_stat[2]

    # scan statistics
    if stat_type != 'linear':
        if type(score).__module__ == 'numpy':
            if normalization == 'schur':
                order = np.argsort(score[idx]**2  / np.diag(Ischur))
            else:
                stat_diag = score[idx]**2 / np.diag(info)[idx]
                stat_diag += score[idx]**2 / np.diag(finfo - info)[idx]
                order = np.argsort(stat_diag)
        else:
            if normalization == 'schur':
                _, order = torch.sort(score[idx]**2 / torch.diag(Ischur))
            else:
                stat_diag = score[idx]**2 / torch.diag(info)[idx]
                stat_diag += score[idx]**2 / torch.diag(finfo - info)[idx]
                _, order = torch.sort(stat_diag)
        for j, p in enumerate(prange):
            if normalization == 'schur':
                new_score = _max_score(int(p), order, score[idx], Ischur)
            else:
                new_score = _max_score(
                    int(p), order, score[idx], info[np.ix_(idx, idx)])
                new_score += _max_score(
                    int(p), order, score[idx], (finfo - info)[np.ix_(idx, idx)])
            new_index = torch.tensor(idx)[order[-int(p):]]
            if stat_type != 'autograd':
                new_stat[1] = _compute_normalized_stat(new_score, int(p), thresh[1][j])
            if stat_type != 'scan':
                new_stat[2] = _compute_normalized_stat(new_score, int(p), thresh[3][j])
            if new_stat[1] > stat[1]:
                stat[1], index[1] = new_stat[1], new_index
            if new_stat[2] > stat[2]:
                stat[2], index[2] = new_stat[2], new_index
    return stat, index


def _update_mean(mean, size, inputs):
    """Update mean in an online fashion."""
    new_size = size + len(inputs)
    if type(mean).__module__ == 'numpy':
        new_mean = size / new_size * mean + np.sum(inputs, axis=0) / new_size
    else:
        new_mean = size / new_size * mean + torch.sum(inputs, axis=0) / new_size
    return new_mean, new_size


def _compute_culinear_stat(score, info, thresh):
    """Compute Culinear statistic."""
    try:
        if type(score).__module__ == 'numpy':
            inv = np.linalg.solve(info, score)
        else:
            inv, _ = torch.solve(score.view(-1, 1), info)
        return score @ inv / thresh
    except (RuntimeError, np.linalg.LinAlgError) as _:
        #inv, _ = torch.solve(sub_score.view(-1, 1), sub_info + 0.1 * torch.eye(self._dim))
        return 0.0
    
    
def conjugate_grad(init_val, grad, max_iter=100, accuracy=1e-7):
    """Solve a quadratic prolem using the conjugate gradient method.

    Parameters
    ----------
    init_val : torch.Tensor
        Initial point.
    grad : callable
        A function outputs the gradient of the quadratic.

    Returns
    -------
    x : torch.Tensor
        The solution to the problem.
    """
    x = init_val
    r = -grad(x)
    b = -grad(torch.zeros_like(x))
    v = r
    for k in range(max_iter):
        Av = grad(v) + b
        vnorm2 = torch.dot(v, Av)
        rnorm2 = torch.dot(r, r)
        t = rnorm2 / vnorm2
        x = x + t * v  # update x in direction v
        r = r - t * Av
        if torch.norm(r) < accuracy:
            break
        beta = torch.dot(r, r) / rnorm2
        v = r + beta * v
    return x


##########################################################################
# Miscellaneous
##########################################################################

def _exceptions_handling(n, d, alpha, lag, idx, prange, trange,
                         stat_type, computation='standard'):
    """Handle exceptions and assign default values"""
    if stat_type not in ['linear', 'scan', 'autograd', 'all']:
        raise NameError("Invalid type of statistic. Only 'linear', 'scan', 'autograd', and 'all' are implemented.")
    
    if computation not in ['conjugate', 'standard']:
        raise NameError("Invalid computation strategy. Only 'conjugate' and 'standard' are implemented.")

    if isinstance(alpha, float):
        alphas = [alpha, alpha, alpha/2, alpha/2]
    elif isinstance(alpha, list) & (len(alpha) == 2):
        alphas = [sum(alpha), sum(alpha), *alpha]
    else:
        raise ValueError("Invalid alpha. It must be a float or a list of length two.")

    if idx is None: idx = range(d)
    if prange is None:
        prange = range(1, min([int(np.sqrt(d)), len(idx)]) + 1)
    elif min(prange) < 1 or max(prange) > len(idx):
        raise ValueError("Invalid prange. It must be within {1, ..., len(idx)}.")

    _lag = lag or 0  # if lag is None, set it to 0
    if trange is None:
        trange = range(int(n / 10) + _lag, int(n * 9 / 10))
    elif min(trange) < _lag or max(trange) > n:
        raise ValueError("Invalid trange. It must be within {lag, ..., n}.")
    return alphas, idx, prange, trange


def _update_res(new_stat, stat, new_index, index, t, tau):
    if new_stat[0] > stat[0]:
        stat[0], index[0], tau[0] = new_stat[0], new_index[0], t
    if new_stat[1] > stat[1]:
        stat[1], index[1], tau[1] = new_stat[1], new_index[1], t
    if new_stat[2] > stat[2]:
        stat[2], index[2], tau[2] = new_stat[2], new_index[2], t
    return stat, index, tau


def _return_results(stat, index, tau, stat_type):
    if stat_type == 'linear':
        return stat[0], tau[0]
    elif stat_type == 'scan':
        return stat[1], tau[1], index[1]
    elif stat_type == 'autograd':
        return stat[2], tau[2], index[2]
    else:
        return stat, tau, index
