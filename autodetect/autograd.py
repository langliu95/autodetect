"""Module for autograd-test.

Author: Lang Liu
Date: 06/10/2019
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
from .utils import _return_results
from .utils import _update_res


class AutogradTest(object):
    """A class for autograd-test for models inherited from :class:`torch.nn.Module`.

    Define and train your model using :class:`torch.nn.Module`
    before calling this class.

    .. note::
        This class treats all model parameters as a single parameter vector to
        compute derivatives of the log-likelihood function.
        This parameter vector is obtained by iterating over parameters of your
        pre-trained model and reshaping each of them to a vector by row.

        The log-likelihood function is closely related to loss function in
        machine learning literature. Negative log-likelihood functions can be
        used as loss functions; while some loss functions have corresponding
        log-likelihood functions (such as mean square error versus
        log-likelihood of Gaussian).

        For latent variable models, computing the score and information may be
        time-consuming. In that case you should consider implementing the
        calculation using specifically designed algorithms (see, for instance, the
        class :class:`AutogradHmm`).

    Parameters
    ----------
    pretrained_model: torch.nn.Module
        A pre-trained model inherited from :class:`torch.nn.Module`.
    loglike: function
        ``loglike(outputs, targets)`` is the log-likelihood of model parameters
        given ``outputs`` and ``targets``.
    """

    def __init__(self, pretrained_model, loglike):
        super(AutogradTest, self).__init__()
        self._dim = sum([par.numel() for par in pretrained_model.parameters() if
                        par.requires_grad])
        self._model = copy.deepcopy(pretrained_model)
        self._loglike = loglike
        self._max_iter = 2 * self._dim
        self._accuracy = 1e-7
        self._indices = range(self._dim)

    def log_likelihood(self, inputs, targets):
        """Compute log-likelihood."""
        outputs = self._model(inputs).view(-1)
        return self._loglike(outputs, targets)

    def gradients(self):
        """Get gradient of model parameters.

        Returns an 1D ``Tensor`` contains the gradient of parameters.
        """
        grads = [p.grad.detach().view(-1) for p in self._model.parameters()]
        return torch.cat(grads)[self._indices]

    def zero_grad(self):
        """Set gradient of the model to zero."""
        self._model.zero_grad()
        
    def score_func(self, inputs, targets):
        """Compute score function."""
        self.zero_grad()
        like = self.log_likelihood(inputs, targets)
        like.backward()
        return self.gradients()[self._indices]
    
    def _score_func(self, inputs, targets):
        obj = self.log_likelihood(inputs, targets)
        grads = grad(obj, self._model.parameters(), create_graph=True)
        grads = torch.cat([g.view(-1) for g in grads])[self._indices]
        return grads

    def information(self, inputs, targets):
        """Compute score function and information matrix."""
        self.zero_grad()
        grads = self._score_func(inputs, targets)
        dim = len(self._indices)
        info = torch.zeros((dim, dim))
        ident = torch.eye(dim)
        for i in range(dim):
            grads.backward(ident[i, :], retain_graph=True)
            info[i, :] = -self.gradients()
            self.zero_grad()
        return grads.detach(), info.detach()
    
    def vec_info_prod(self, inputs, targets, vec, create_graph=True):
        """Compute vector-information product."""
        self.zero_grad()
        if create_graph:
            grads = self._score_func(inputs, targets)
        else:
            grads = self._grads[self._indices]
        obj = -grads @ vec
        obj.backward(retain_graph=True)
        return self.gradients()

    # TODO: change the full information method in the same format by storing info
    def inv_info_vec_prod(self, inputs, targets, vec,
                          max_iter=100, accuracy=1e-7):
        """Compute inverse-information-vector product."""
        npar = len(vec)
        if npar == 1:
            info = self.vec_info_prod(inputs, targets, torch.tensor([1.0]))
            return (vec / info).detach()
        
        self.zero_grad()
        self._grads = self._score_func(inputs, targets)
        
        def quad_grad(x):
            r = self.vec_info_prod(inputs, targets, x, create_graph=False) - vec
            return r
        
        init = torch.randn(npar)
        x = conjugate_grad(init, quad_grad, max_iter=max_iter,
                           accuracy=accuracy)

        self._grads = None
        return x.detach()

    def compute_stats(self, inputs, targets, alpha=0.05, lag=0, idx=None,
                      prange=None, trange=None, stat_type='autograd',
                      computation='standard', normalization='schur',
                      max_iter=None, accuracy=1e-7):
        """Compute test statistics.

        This function performs score-based hypothesis tests to detect the
        existence of a change in machine learning systems as they learn from
        a continuous, possibly evolving, stream of data.
        Three tests are implemented: the linear test, the scan test, and the autograd-test. The
        linear statistic is the maximum score statistic over all possible locations of
        change. The scan statistic is the maximum score statistic over all possible
        locations of change, and over all possible subsets of parameters in which change occurs.

        Parameters
        ----------
        inputs: torch.Tensor, shape (size, dim)
        targets: torch.Tensor, shape (size, \*)
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
            inputs.size(0), self._dim, alpha, lag, idx, prange, trange,
            stat_type, computation)
        
        self._configuration(
            lag, idx, prange, trange, stat_type, max_iter, accuracy)
        
        # computes the score and information matrix once
        score, info = self.information(inputs, targets)
        # computes thresholds once
        thresh = compute_thresholds(
            len(idx), max([1, max(trange) - min(trange)]),
            alpha, prange, stat_type)
        
        if computation == 'conjugate':
            res = self._conjugate_stats(inputs, targets, score, info, thresh)
        else:
            res = self._standard_stats(inputs, targets, score, info,
                                       thresh, normalization)
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
    
    def _standard_stats(self, inputs, targets, score, info, thresh, normalization):
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
                _score, _info = self.information(
                    inputs[(lo - self._lag):t], targets[(lo - self._lag):t])
                score_t -= _score
                info_t -= _info
            else:
                score_t, info_t = self.information(inputs[:t], targets[:t])
                score_t = score - score_t
                info_t = info - info_t
            new_stat, new_index = _compute_stats(
                self._prange, self._idx, score_t, info_t, Iinv,
                thresh, self._stat_type, normalization, finfo=info)
            stat, index, tau = _update_res(
                new_stat, stat, new_index, index, t, tau)
        return _return_results(stat, index, tau, self._stat_type)
    
    def _conjugate_stats(self, inputs, targets, score, info, thresh):
        """Compute test statistics using the conjugate gradient algorithm."""
        idx = self._idx
        # computes test statistic
        stat, new_stat = torch.zeros(3), torch.zeros(3)
        tau = np.array([0, 0, 0])
        index = [idx, np.arange(max(self._prange)), idx]
        new_index = list(index)
        # compute the diagonal information in the middle once
        half = int(len(inputs)/2)
        info_diag = self._compute_info_diag(
            inputs[:half], targets[:half], info, idx)
        for t in self._trange:
            self._grads = self._score_func(inputs[:t], targets[:t])
            score_t = self._grads.detach()
            score_t = score - score_t
            # linear statistics
            if self._stat_type in ['linear', 'autograd', 'all']:
                _stat = self._conjugate_raw_stat(
                    inputs[:t], targets[:t], score_t[idx],
                    info[np.ix_(idx, idx)], idx)
                new_stat[0] = _compute_normalized_stat(_stat, len(idx), thresh[0])
                new_stat[2] = _compute_normalized_stat(_stat, len(idx), thresh[2])
                new_index[2] = idx
            # scan statistics ([I_{1:tau}]_{T, T}^{-1} + [I_{tau+1:n}]_{T, T}^{-1})
            if self._stat_type in ['scan', 'autograd', 'all']:
                # order = self._conjugate_sort_diag(
                #     inputs[:t], targets[:t], score_t[idx], info[np.ix_(idx, idx)])
                order = self._sort_diag(score_t[idx], info_diag)
                _stat, _index = self._conjugate_scan_stat(
                    inputs[:t], targets[:t], score_t, info, thresh, order)
                new_stat[1], new_index[1] = _stat[1], _index[1]
                if _stat[2] > new_stat[2]:
                    new_stat[2], new_index[2] = _stat[2], _index[2]
            
            stat, index, tau = _update_res(
                new_stat, stat, new_index, index, t, tau)

        self._grads = None
        return _return_results(stat, index, tau, self._stat_type)
    
    def _conjugate_raw_stat(self, inputs, targets, score, full_info, indices=None):
        """Compute the unnormalized score statistic at a given changepoint."""
        self.zero_grad()
        if indices is not None:
            self._indices = indices
        stat = score @ self._inv_info_vec_prod(inputs, targets, score)
        stat += score @ self._inv_info_vec_prod(
            inputs, targets, score, info=full_info)
        self._indices = range(self._dim)
        return stat
    
    def _conjugate_scan_stat(self, inputs, targets, score, full_info,
                             thresh, order):
        """Compute the scan statistic given the ordering and the changepoint."""
        stat = torch.zeros(3)
        index = [self._idx, np.arange(max(self._prange)), self._idx]
        for j, p in enumerate(self._prange):
            _idx = torch.tensor(self._idx)[order[-p:]]
            _stat = self._conjugate_raw_stat(
                inputs, targets, score[_idx], full_info[np.ix_(_idx, _idx)], _idx)
            if self._stat_type in ['scan', 'all']:
                new_stat = _compute_normalized_stat(_stat, int(p), thresh[1][j])
                if new_stat > stat[1]:
                    stat[1], index[1] = new_stat, _idx
            if self._stat_type in ['autograd', 'all']:
                new_stat = _compute_normalized_stat(_stat, int(p), thresh[3][j])
                if new_stat > stat[2]:
                    stat[2], index[2] = new_stat, _idx
        return stat, index
    
    def _compute_info_diag(self, inputs, targets, info, idx):
        """Compute the diagonal information matrices."""
        _, half_info = self.information(inputs, targets)
        info_diag = [
            torch.diag(half_info)[idx], torch.diag(info - half_info)[idx]]
        return info_diag
    
    def _sort_diag(self, score, info_diag):
        """Sort the score statistic according to the diagonal terms."""
        stat_diag = score**2 / info_diag[0]
        stat_diag += score**2 / info_diag[1]
        _, order = torch.sort(stat_diag)
        return order

    def _inv_info_vec_prod(self, inputs, targets, vec, info=None):
        """Compute inverse-information-vector product."""
        npar = len(vec)
        if npar == 1:
            diag_info = self.vec_info_prod(inputs, targets, torch.tensor([1.0]))
            if info is not None:
                diag_info = info[0] - diag_info
            return vec / diag_info
        
        if info is None:  # for I_{1:tau}
            def quad_grad(x):
                r = self.vec_info_prod(
                    inputs, targets, x, create_graph=False) - vec
                return r
        else:  # for I_{1:n} - I_{1:tau}
            def quad_grad(x):
                r = info @ x - self.vec_info_prod(
                    inputs, targets, x, create_graph=False) - vec
                return r
        
        init = torch.randn(npar)
        x = conjugate_grad(init, quad_grad, max_iter=self._max_iter,
                           accuracy=self._accuracy)
        return x.detach()
