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

def _compute_inv(ident, info, const=0.0):
    """Compute inverse of observed information."""
    while True:
        try:
            Iinv, _ = torch.gesv(ident, info + const * ident)
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
    index: array-like
        Indices of parameters which attain the maximal statistic.
    """

    index = order[-p:]
    sub_score = score[index]  # TODO: might need to add .view(-1)
    sub_Ischur = Ischur[np.ix_(index, index)]
    try:
        if type(score).__module__ == 'numpy':
            inv = np.linalg.solve(sub_Ischur, sub_score)
        else:
            inv, _ = torch.gesv(sub_score, sub_Ischur)
        stat = sub_score @ inv
    except (RuntimeError, np.linalg.LinAlgError) as _:
        # print("Conditional information matrix is not invertible.")
        # print("Adding a scalar matrix for an approximation.")
        # inv, _ = torch.gesv(sub_score, sub_Ischur + 0.1*torch.eye(p))
        stat = False  # if sub_Ischur is ill-conditioned, set the stat to False
    return stat, index


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


def _compute_stats(prange, idx, score, info, Iinv, thresh, stat_type='all'):
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
    Iinv : torch.Tensor, shape (dim, dim)
        Inverse of the full observed information.
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
    Ischur = info[np.ix_(idx, idx)] - info[idx, :] @ Iinv @ info[:, idx]

    # linear statistics
    new_score, _ = _max_score(0, range(dim), score[idx], Ischur)
    new_stat[0] = _compute_normalized_stat(new_score, dim, thresh[0])
    new_stat[2] = _compute_normalized_stat(new_score, dim, thresh[2])
    if new_stat[0] > stat[0]: stat[0] = new_stat[0]
    if new_stat[2] > stat[2]: stat[2] = new_stat[2]
    # scan statistics
    if stat_type != 'linear':
        if type(score).__module__ == 'numpy':
            order = np.argsort(score[idx]**2  / np.diag(Ischur))
        else:
            _, order = torch.sort(score[idx]**2 / torch.diag(Ischur))
        for j, p in enumerate(prange):
            new_score, new_index = _max_score(int(p), order, score[idx], Ischur)
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
            inv, _ = torch.gesv(score, info)
        return score @ inv / thresh
    except (RuntimeError, np.linalg.LinAlgError) as _:
        #inv, _ = torch.gesv(sub_score, sub_info + 0.1 * torch.eye(self._dim))
        return 0.0


##########################################################################
# Miscellaneous
##########################################################################

def _exceptions_handling(n, d, alpha, lag, idx, prange, trange, stat_type):
    """Handle exceptions and assign default values"""
    if stat_type not in ['linear', 'scan', 'autograd', 'all']:
        raise NameError("Invalid type of statistic. Only 'linear', 'scan', 'autograd', and 'all' are implemented.")

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