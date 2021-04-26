"""Module for autograd-test.

Author: Lang Liu
Date: 04/08/2021
"""

from __future__ import absolute_import, division, print_function
import copy
import numpy as np

import torch
from torch.autograd import grad

from .utils import compute_thresholds
from .utils import conjugate_grad
from .utils import _compute_inv
from .utils import _compute_normalized_stat
from .utils import _compute_stats
from .utils import _exceptions_handling
from .utils import _get_batch
from .utils import _return_results
from .utils import _update_res


class AutogradFunc(object):
    """A class for autograd-test for models with a log-likelihood function.

    Implement the log-likelihood function using :class:`torch`
    before calling this class.

    Parameters
    ----------
    loglike: callable
        ``loglike(pars, obs)`` is the log-likelihood of model parameters
        given ``obs``.
    pars: list
        List of model parameters.
    """

    def __init__(self, loglike, pars):
        super(AutogradFunc, self).__init__()
        self._loglike = loglike
        self._dim = sum([p.numel() for p in pars])
        self._pars = [p.clone().detach().requires_grad_(True) for p in pars]
        self._max_iter = 2 * self._dim
        self._accuracy = 1e-7
        self._indices = range(self._dim)

    def log_likelihood(self, obs):
        """Compute log-likelihood."""
        return self._loglike(self._pars, obs)

    def gradients(self):
        """Get gradient of model parameters."""
        grads = [p.grad.detach().view(-1) for p in self._pars]
        return torch.cat(grads)[self._indices]

    def zero_grad(self):
        """Set gradient of model parameters to zero."""
        for p in self._pars:
            if p.grad is not None:
                p.grad.data.zero_()
        
    def score_func(self, obs):
        """Compute score function."""
        self.zero_grad()
        obj = self.log_likelihood(obs)
        obj.backward()
        return self.gradients()[self._indices]
    
    def _score_func(self, obs):
        """Compute score function."""
        obj = self.log_likelihood(obs)
        grads = grad(obj, self._pars, create_graph=True)
        grads = torch.cat([g.view(-1) for g in grads])[self._indices]
        return grads

    def information(self, obs):
        """Compute score function and information matrix."""
        self.zero_grad()
        grads = self._score_func(obs)
        dim = len(self._indices)
        info = torch.zeros((dim, dim))
        ident = torch.eye(dim)
        for i in range(dim):
            grads.backward(ident[i, :], retain_graph=True)
            info[i, :] = -self.gradients()
            self.zero_grad()
        return grads.detach(), info.detach()
    
    def vec_info_prod(self, obs, vec, create_graph=True):
        """Compute vector-information product."""
        self.zero_grad()
        if create_graph:
            grads = self._score_func(obs)
        else:
            grads = self._grads[self._indices]
        loss = -grads @ vec
        loss.backward(retain_graph=True)
        return self.gradients()

    def inv_info_vec_prod(self, obs, vec, max_iter=100, accuracy=1e-7):
        """Compute inverse-information-vector product."""
        npar = len(vec)
        if npar == 1:
            diag_info = self.vec_info_prod(obs, torch.tensor([1.0]))
            return vec / diag_info
        
        self.zero_grad()
        self._grads = self._score_func(obs)
        
        def quad_grad(x):
            r = self.vec_info_prod(obs, x, create_graph=False) - vec
            return r
        
        init = torch.randn(npar)
        x = conjugate_grad(init, quad_grad, max_iter=max_iter,
                           accuracy=accuracy)
        
        self._grads = None
        return x.detach()

    def compute_stats(self, obs, alpha=0.05, lag=0, idx=None, prange=None,
                      trange=None, stat_type='autograd', computation='standard',
                      normalization='schur', max_iter=None, accuracy=1e-7):
        """Compute test statistics.

        This function performs score-based hypothesis tests to detect the
        existence of a change in machine learning systems as they learn from
        a continuous, possibly evolving, stream of data.
        Three tests are implemented: the linear test, the scan test, and the
        autograd-test. The linear statistic is the maximum score statistic over
        all possible locations of change. The scan statistic is the maximum
        score statistic over all possible locations of change, and over all
        possible subsets of parameters in which change occurs.

        Parameters
        ----------
        obs: list of torch.Tensors
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
        computation: str, optional
            Strategy to compute the test statistic. If ``'conjugate'``, then use
            the conjugate gradient algorithm to compute inverse-Hessian-vector
            product; if ``'standard'``, then use the full Fisher information to
            compute the statistic. We recommend ``'standard'`` if data are
            independent and ``'conjugate'`` otherwise. Default is ``'standard'``.
        normalization: str, optional
            Normalization matrix. If ``'schur'``, then use the Schur complement
            as the normalization matrix; if ``'additive'``, then use
            :math:`I_{1:\\tau}^{-1} + I_{\\tau+1:n}^{-1}`. Default is ``'schur'``.
        max_iter: int, optional
            Maximum number of iterations in the conjugate gradient algorithm.
            Default is `None`, which will be set to ``2 * dim``.
        accuracy: float, optional
            Accuracy in the conjugate gradient algorithm.
            Default is `1e-7`.

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
            If ``stat_type`` is not in ``['linear', 'scan', 'autograd', 'all']``;
            or if ``computation`` is not in ``['conjugate, 'standard']``.
        ValueError
            If ``alpha`` is not an instance of ``float`` or ``list``; or if ``prange``
            is not within ``range(1, len(idx)+1)``; or if ``trange`` is not within
            ``range(lag, size)``.
        """

        alpha, idx, prange, trange = _exceptions_handling(
            obs[0].size(0), self._dim, alpha, lag, idx, prange, trange,
            stat_type, computation)
        
        self._configuration(
            lag, idx, prange, trange, stat_type, max_iter, accuracy)
        
        # computes the score and information matrix once
        score, info = self.information(obs)
        # computes thresholds once
        thresh = compute_thresholds(
            len(idx), max([1, max(trange) - min(trange)]),
            alpha, prange, stat_type)
        
        if computation == 'conjugate':
            res = self._conjugate_stats(obs, score, info, thresh)
        else:
            res = self._standard_stats(obs, score, info, thresh, normalization)
        return res
    
    def _configuration(self, lag, idx, prange, trange,
                       stat_type, max_iter, accuracy):
        """Configure class attributes."""
        self._lag = lag
        self._idx = idx
        self._prange = prange
        self._trange = trange
        self._stat_type = stat_type
        if max_iter is not None:
            self._max_iter = max_iter
        else:
            self._max_iter = 2 * len(idx)
        self._accuracy = accuracy
    
    def _standard_stats(self, obs, score, info, thresh, normalization):
        """Compute test statistics using the full Fisher information."""
        # computes the inverse of information matrix once
        Iinv = torch.eye(self._dim)
        if normalization == 'schur':
            Iinv = _compute_inv(Iinv, info)
        # computes test statistic
        stat = torch.zeros(3)
        tau = np.array([0, 0, 0])
        index = [self._idx, np.arange(max(self._prange)), self._idx]
        score_t = torch.clone(score)
        info_t = torch.clone(info)
        for i, t in enumerate(self._trange):
            if self._lag is not None:
                lo = self._trange[i-1] if i > 0 else self._lag
                _score, _info = self.information(_get_batch(obs, lo-self._lag, t))
                score_t -= _score
                info_t -= _info
            else:
                score_t, info_t = self.information(_get_batch(obs, 0, t))
                score_t = score - score_t
                info_t = info - info_t
            new_stat, new_index = _compute_stats(
                self._prange, self._idx, score_t, info_t, Iinv,
                thresh, self._stat_type, normalization, finfo=info)
            stat, index, tau = _update_res(
                new_stat, stat, new_index, index, t, tau)
        return _return_results(stat, index, tau, self._stat_type)
    
    def _conjugate_stats(self, obs, score, info, thresh):
        """Compute test statistics using the conjugate gradient method."""
        idx = self._idx
        # computes test statistic
        stat, new_stat = torch.zeros(3), torch.zeros(3)
        tau = np.array([0, 0, 0])
        index = [idx, np.arange(max(self._prange)), idx]
        new_index = list(index)
        # compute the diagonal information in the middle once
        info_diag = self._compute_info_diag(obs, int(len(obs[0])/2), info, idx)
        for t in self._trange:
            batch = _get_batch(obs, 0, t)
            self._grads = self._score_func(batch)
            score_t = self._grads.detach()
            score_t = score - score_t
            # linear statistics
            if self._stat_type in ['linear', 'autograd', 'all']:
                _stat = self._conjugate_raw_stat(
                    batch, score_t[idx], info[np.ix_(idx, idx)], idx)
                new_stat[0] = _compute_normalized_stat(_stat, len(idx), thresh[0])
                new_stat[2] = _compute_normalized_stat(_stat, len(idx), thresh[2])
                new_index[2] = idx
            # scan statistics ([I_{1:tau}]_{T, T}^{-1} + [I_{tau+1:n}]_{T, T}^{-1})
            if self._stat_type in ['scan', 'autograd', 'all']:
                order = self._sort_diag(score_t[idx], info_diag)
                _stat, _index = self._conjugate_scan_stat(
                    batch, score_t, info, thresh, order)
                new_stat[1], new_index[1] = _stat[1], _index[1]
                if _stat[2] > new_stat[2]:
                    new_stat[2], new_index[2] = _stat[2], _index[2]
            
            stat, index, tau = _update_res(
                new_stat, stat, new_index, index, t, tau)

        self._grads = None
        return _return_results(stat, index, tau, self._stat_type)
    
    def _conjugate_raw_stat(self, obs, score, full_info, indices=None):
        """Compute the unnormalized score statistic at a given changepoint."""
        self.zero_grad()
        if indices is not None:
            self._indices = indices
        stat = score @ self._inv_info_vec_prod(obs, score)
        stat += score @ self._inv_info_vec_prod(obs, score, info=full_info)
        self._indices = range(self._dim)
        return stat
    
    def _conjugate_scan_stat(self, obs, score, full_info, thresh, order):
        """Compute the scan statistic given the ordering and the changepoint."""
        stat = torch.zeros(3)
        index = [self._idx, np.arange(max(self._prange)), self._idx]
        for j, p in enumerate(self._prange):
            _idx = torch.tensor(self._idx)[order[-p:]]
            _stat = self._conjugate_raw_stat(
                obs, score[_idx], full_info[np.ix_(_idx, _idx)], _idx)
            if self._stat_type in ['scan', 'all']:
                new_stat = _compute_normalized_stat(_stat, int(p), thresh[1][j])
                if new_stat > stat[1]:
                    stat[1], index[1] = new_stat, _idx
            if self._stat_type in ['autograd', 'all']:
                new_stat = _compute_normalized_stat(_stat, int(p), thresh[3][j])
                if new_stat > stat[2]:
                    stat[2], index[2] = new_stat, _idx
        return stat, index
    
    def _compute_info_diag(self, obs, t, info, idx):
        """Compute the diagonal information matrices at time ``t``."""
        batch = _get_batch(obs, 0, t)
        _, half_info = self.information(batch)
        info_diag = [
            torch.diag(half_info)[idx], torch.diag(info - half_info)[idx]]
        return info_diag
    
    def _sort_diag(self, score, info_diag):
        """Sort the score statistic according to the diagonal terms."""
        # old_info_diag = torch.zeros(len(self._idx))
        # for i, ind in enumerate(self._idx):
        #     vec = torch.zeros(self._dim)
        #     vec[ind] = 1
        #     old_info_diag[i] = self.vec_info_prod(obs, vec)[ind]
        # stat_diag = score**2 / old_info_diag
        # stat_diag += score**2 / (torch.diag(full_info) - old_info_diag)
        stat_diag = score**2 / info_diag[0]
        stat_diag += score**2 / info_diag[1]
        _, order = torch.sort(stat_diag)
        return order

    def _inv_info_vec_prod(self, obs, vec, info=None):
        """Compute inverse-information-vector product."""
        npar = len(vec)
        if npar == 1:
            diag_info = self.vec_info_prod(obs, torch.tensor([1.0]))
            if info is not None:
                diag_info = info[0] - diag_info
            return vec / diag_info
        
        if info is None:  # for I_{1:tau}
            def quad_grad(x):
                r = self.vec_info_prod(obs, x, create_graph=False) - vec
                return r
        else:  # for I_{1:n} - I_{1:tau}
            def quad_grad(x):
                r = info @ x - self.vec_info_prod(obs, x, create_graph=False) - vec
                return r
        
        init = torch.randn(npar)
        x = conjugate_grad(init, quad_grad, max_iter=self._max_iter,
                           accuracy=self._accuracy)
        return x.detach()
