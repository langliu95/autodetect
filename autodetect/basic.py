"""Functions for autograd-test (stand-alone models).

Author: Lang Liu
Date: 06/10/2019
"""

from __future__ import absolute_import, division, print_function
import numpy as np

import torch
from torch.autograd import grad

from .utils import compute_thresholds
from .utils import _compute_inv
from .utils import _compute_stats
from .utils import _exceptions_handling
from .utils import _return_results
from .utils import _update_res


def _score_information(theta, obs, loglike):
    """Computes score function and information matrix.

    The information matrix here is computed by the outer product of the score function.
    """
    score = torch.zeros(len(theta))
    info = torch.zeros((len(theta), len(theta)))
    for ob in obs:
        like = loglike(theta, ob.view(1, -1))
        temp = grad(like, theta)[0]
        score += temp
        info += torch.ger(temp, temp)
    return score.detach(), info.detach()


def score_function(theta, obs, loglike):
    """Computes score function.

    Parameters
    ----------
    theta: torch.Tensor, shape (dim,)
        Values of parameters at which score and information are computed.
    obs: torch.Tensor, shape (size, dim)
        Observations.
    loglike: function
        ``loglike(theta, obs)`` is the log-likelihood of ``theta`` given ``obs``.
    """
    like = loglike(theta, obs)
    score = grad(like, theta, create_graph=True)[0]
    return score.detach()


def information(theta, obs, loglike, ident=None):
    """Computes score function and information matrix.

    Parameters
    ----------
    theta: torch.Tensor, shape (dim,)
        Values of parameters at which score and information are computed.
    obs: torch.Tensor, shape (size, dim)
        Observations.
    loglike: function
        ``loglike(theta, obs)`` is the log-likelihood of ``theta`` given ``obs``.
    ident: torch.Tensor, shape (dim, dim), optional
        Identity matrix. Default is ``None``.
    """

    d = len(theta)
    like = loglike(theta, obs)
    score = grad(like, theta, create_graph=True)[0]

    info = torch.zeros((d, d))
    if ident is None: ident = torch.eye(d)
    for i in range(d):
        score.backward(ident[i, :], retain_graph=True)
        info[i, :] = -theta.grad
        theta.grad.data.zero_()
    return score.detach(), info.detach()


def autograd_test(theta, obs, loglike, alpha=0.05, lag=0, idx=None, prange=None,
    trange=None, stat_type='autograd'):
    """Computes autograd-test statistics.

    This function performs score-based hypothesis tests to detect the existence of
    a change in machine learning systems as they learn from
    a continuous, possibly evolving, stream of data.
    Three tests are implemented: the linear test, the scan test, and the autograd-test. The
    linear statistic is the maximum score statistic over all possible locations of
    change. The scan statistic is the maximum score statistic over all possible
    locations of change, and over all possible subsets of parameters in which change occurs.

    .. note::
        The log-likelihood function ``loglike`` needs to be implemented in
        :class:`PyTorch`.

    Parameters
    ----------
    theta: torch.Tensor, shape (dim,)
        Maximum likelihood estimator of model parameters under null hypothesis
        (no change exists).
    obs: torch.Tensor, shape (size, dim)
        Observations.
    loglike: function
        ``loglike(theta, obs)`` is the log-likelihood of ``theta`` given ``obs``.
    alpha: double or list, optional
        Significance level(s). For the autograd-test it should be a list of length two,
        where the first element is the significance level for the linear statistic and
        the second is for the scan statistic. Default is 0.05.
    lag: int, optional
        Order of Markovian dependency. The distribution of ``obs[k]`` only
        depends on ``obs[(k-lag):k]``. Use ``None`` to represent
        non-Markovian dependency. Default is 0.
    idx: array-like, optional
        Indices of parameters of interest (the rest parameters are considered constants)
        in the parameter vector.
        Default is ``None``, which will be set to ``range(dim)``.
    prange: array-like, optional
        Change cardinality set over which the scan statistic is maximized.
        Default is ``None``,
        which will be set to ``range(1, min([int(np.sqrt(d)), len(idx)]) + 1)``.
    trange: array-like, optional
        Change location set over which the statistic is maximized. Default is ``None``,
        which will be set to ``range(int(n / 10) + lag, int(n * 9 / 10))``.
    stat_type: str, optional
        Type of statistic that is computed. It can take values in ``['linear', 'scan',
        'autograd', 'all']``, where ``'all'`` indicates calculating all of them. Default is ``'autograd'``.

    Returns
    -------
    stat: torch.Tensor
        Test statistic at level ``alpha``. Reject null if it is larger than 1.
    tau: int
        Location of changepoint corresponds to the test statistic.
    index: array-like
        Indices of parameters correspond to the test statistic. It will be omitted for the linear test.

    Raises
    ------
    NameError
        If ``stat_type`` is not in ``['linear', 'scan', 'autograd', 'all']``.
    ValueError
        If ``alpha`` is not an instance of ``float`` or ``list``; or if ``prange``
        is not within ``range(1, len(idx)+1)``; or if ``trange`` is not within
        ``range(lag, size)``.
    """
    # creates results for [linear, scan, autograd],
    # then returns the one(s) based on ``stat_type``.
    theta.requires_grad_(True)
    alpha, idx, prange, trange = _exceptions_handling(len(obs), len(theta), alpha,
        lag, idx, prange, trange, stat_type)
    # computes the inverse of information matrix once
    ident = torch.eye(len(theta))
    score, info = information(theta, obs, loglike, ident)
    Iinv = _compute_inv(ident, info)
    # computes thresholds once
    thresh = compute_thresholds(len(idx), max([1, max(trange) - min(trange)]),
        alpha, prange, stat_type)
    # computes test statistic
    stat = torch.zeros(3)
    tau = np.array([0, 0, 0])
    index = [idx, np.arange(max(prange)), idx]
    if lag is not None:
        for lo, hi in zip([lag, *trange[:-1]], trange):
            # computes conditional score and information
            score_t, info_t = information(theta, obs[(lo-lag):hi], loglike, ident)
            score -= score_t
            info -= info_t
            new_stat, new_index = _compute_stats(prange, idx, score, info, Iinv,
                thresh, stat_type)
            stat, index, tau = _update_res(new_stat, stat, new_index, index, hi, tau)
    else:  # non-Markovian dependency
        for t in trange:
            score_t, info_t = information(theta, obs[:t], loglike, ident)
            score_t = score - score_t
            info_t = info - info_t
            new_stat, new_index = _compute_stats(prange, idx, score, info, Iinv,
                thresh, stat_type)
            stat, index, tau = _update_res(new_stat, stat, new_index, index, t, tau)
    return _return_results(stat, index, tau, stat_type)


def _arma_einformation(theta, obs, p, q, ident=None):
    """Computes score function and information matrix for error term.

    The observations must be one-dimensional.

    Parameters
    ----------
    theta: torch.Tensor, shape (dim,)
        Values of parameters at which score and information are computed.
    obs: torch.Tensor, shape (size, 2)
        The first column is observations and the second one is errors.
    p, q: int
        Order of ARMA model.
    ident: torch.Tensor, shape (dim, dim), optional
        Identity matrix. Default is ``None``.
    """

    x, e = obs[:, 0], obs[:, 1]
    d = len(theta)
    if ident is None: ident = torch.eye(d)
    score = torch.zeros((len(x), d))
    info = torch.zeros((len(x), d, d))
    for t in range(max(p, q), len(x)):
        # computes errors
        inv_idx = torch.arange(t-1, t-p-1, -1).long()
        inv_ide = torch.arange(t-1, t-q-1, -1).long()
        error = x[t] - torch.sum(theta[:p] * x[inv_idx]) -\
            torch.sum(theta[p:(p+q)] * e[inv_ide])
        e[t] = error.detach()
        # computes score
        score_t = grad(error, theta)[0].detach()
        score[t] = score_t - torch.sum(theta[p:(p+q), np.newaxis].detach() * score[inv_ide], 0)
        # computes information
        info[t] = -torch.sum(theta[p:(p+q), np.newaxis, np.newaxis].detach() * info[inv_ide], 0)
        for i in range(q):
            info[t, :, i+p] += score[t-i-1]
    return score.detach(), info.detach()


def arma_information(error, escore, einfo, sig2):
    """Computes the score and information for ARMA based on the ones of errors.
    """
    score = -(torch.sum(error[:, np.newaxis] * escore, 0)) / sig2
    info = (escore.transpose(0, 1) @ escore -
            torch.sum(error[:, np.newaxis, np.newaxis] * einfo, 0)) / sig2
    return score, info


def autograd_arma(theta, sig2, obs, p, q, alpha=0.05, idx=None, prange=None, trange=None,
    stat_type='autograd'):
    """Computes autograd-test statistics for autoregressive--moving-average model.

    This function performs score-based hypothesis tests to detect the existence of
    a change in an autoregressive--moving-average model as
    it learns from a continuous, possibly evolving, stream of data.
    Three tests are implemented: the linear test, the scan test, and the autograd-test. The
    linear statistic is the maximum score statistic over all possible locations of
    change. The scan statistic is the maximum score statistic over all possible
    locations of change, and over all possible subsets of parameters in which change
    occurs (for more details see). TODO: add reference.

    Parameters
    ----------
    theta: torch.Tensor, shape (dim,)
        Conditional maximum likelihood estimator of model parameters under null
        hypothesis (condition on the first ``max(p, q)`` observations).
    sig2: torch.Tensor
        Variance of the residuals.
    obs: torch.Tensor, shape (size, 2)
        The first column is observations and the second one is errors.
    p, q: int
        Order of ARMA model.
    alpha: double or list, optional
        Significance level(s). For the autograd-test it should be a list of length two,
        where the first element is the significance level for the linear statistic and
        the second is for the scan statistic. Default is 0.05.
    idx: array-like, optional
        Indices of parameters of interest (the rest parameters are considered constants)
        in the parameter vector.
        Default is ``None``, which will be set to ``range(dim)``.
    prange: array-like, optional
        Change cardinality set over which the scan statistic is maximized.
        Default is ``None``,
        which will be set to ``range(1, min([int(np.sqrt(d)), len(idx)]) + 1)``.
    trange: array-like, optional
        Change location set over which the statistic is maximized. Default is ``None``,
        which will be set to ``range(int(n / 10) + lag, int(n * 9 / 10))``.
    stat_type: str, optional
        Type of statistic that is computed. It can take values in ``['linear', 'scan',
        'autograd', 'all']``, where ``'all'`` indicates calculating all of them. Default is ``'autograd'``.

    Returns
    -------
    stat: torch.Tensor
        Test statistic at level ``alpha``. Reject null if it is larger than 1.
    tau: int
        Location of changepoint corresponds to the test statistic.
    index: array-like
        Indices of parameters correspond to the test statistic. It will be omitted for the linear test.

    Raises
    ------
    NameError
        If ``stat_type`` is not in ``['linear', 'scan', 'autograd', 'all']``.
    ValueError
        If ``alpha`` is not an instance of ``float`` or ``list``; or if ``prange``
        is not within ``range(1, len(idx)+1)``; or if ``trange`` is not within
        ``range(lag, size)``.
    """

    theta.requires_grad_(True)
    if obs.shape[1] != 2:
        raise ValueError('Only one-dimensional time series is supported.')
    alpha, idx, prange, trange = _exceptions_handling(len(obs), len(theta), alpha,
        max(p, q), idx, prange, trange, stat_type)
    # computes the inverse of information matrix once
    ident = torch.eye(len(theta))
    escore, einfo = _arma_einformation(theta, obs, p, q, ident)
    score, info = arma_information(obs[:, 1], escore, einfo, sig2)
    Iinv = _compute_inv(ident, info)
    # computes thresholds once
    thresh = compute_thresholds(len(idx), max([1, max(trange) - min(trange)]),
        alpha, prange, stat_type)
    # computes test statistic
    stat = torch.zeros(3)
    tau = np.array([0, 0, 0])
    index = [idx, np.arange(max(prange)), idx]
    for t in trange:
        # computes conditional score and information
        score_t, info_t = arma_information(obs[:t, 1], escore[:t], einfo[:t], sig2)
        score_t = score - score_t
        info_t = info - info_t
        new_stat, new_index = _compute_stats(prange, idx, score_t, info_t, Iinv,
            thresh, stat_type)
        stat, index, tau = _update_res(new_stat, stat, new_index, index, t, tau)
    return _return_results(stat, index, tau, stat_type)
