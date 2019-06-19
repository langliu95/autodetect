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
from .utils import _compute_inv
from .utils import _compute_stats
from .utils import _exceptions_handling
from .utils import _return_results
from .utils import _update_res


class AutogradTest(object):
    """A class for autograd-test in models inherited from :class:`torch.nn.Module`.

    You should define and train your model using :class:`torch.nn.Module`
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

        For latent variable models, computing the score and information may be time-consuming.
        In that case you should consider implementing the calculation using specifically designed algorithms (see, for instance, the
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
        self._dim = sum(par.numel() for par in pretrained_model.parameters() if
                        par.requires_grad)
        self._model = copy.deepcopy(pretrained_model)
        self._loglike = loglike

    def log_likelihood(self, inputs, targets):
        """Computes log-likelihood."""
        outputs = self._model(inputs).view(-1)
        return self._loglike(outputs, targets)

    def gradients(self):
        """Gets gradient of model parameters.

        Returns an 1D ``Tensor`` contains the gradient of parameters.
        """
        grads = []
        for par in self._model.parameters():
            grads.append(par.grad[0].view(-1))
        return torch.cat(grads)

    def zero_grad(self):
        """Sets gradient of the model to zero."""
        self._model.zero_grad()

    def information(self, inputs, targets):
        """Computes score function and information matrix.

        .. note::
            This function will set gradients of the model to zero.
        """
        like = self.log_likelihood(inputs, targets)
        grad_tuple = grad(like, self._model.parameters(), create_graph=True)
        grads = []
        info = torch.zeros((self._dim, self._dim))
        ind = 0  # index of parameter
        for g in grad_tuple:
            g = g.view(-1)
            m = g.numel()
            grads.append(g)
            ident = torch.eye(m)
            for j in range(m):
                self._model.zero_grad()
                g.backward(ident[j, :], retain_graph=True)
                info[ind, :] = -self.gradients()
                ind += 1
        return torch.cat(grads).detach(), info.detach()

    def compute_stats(self, inputs, targets, alpha=0.05, lag=0, idx=None, prange=None, trange=None, stat_type='autograd'):
        """Computes test statistics.

        This function performs score-based hypothesis tests to detect the existence of a change in machine learning systems as they learn from
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

        alpha, idx, prange, trange = _exceptions_handling(inputs.size(0),
            self._dim, alpha, lag, idx, prange, trange, stat_type)
        # computes the inverse of information matrix once
        ident = torch.eye(self._dim)
        score, info = self.information(inputs, targets)
        Iinv = _compute_inv(ident, info)
        # computes thresholds once
        thresh = compute_thresholds(len(idx), max([1, max(trange) - min(trange)]),
            alpha, prange, stat_type)
        # computes test statistic
        stat = torch.zeros(3)
        tau = np.array([0, 0, 0])
        index = [idx, np.arange(max(prange)), idx]
        if lag is not None:  # fixed order of Markovian dependence
            for lo, hi in zip([lag, *trange[:-1]], trange):
                score_t, info_t = self.information(inputs[(lo-lag):hi],
                    targets[(lo-lag):hi])
                score -= score_t
                info -= info_t
                new_stat, new_index = _compute_stats(prange, idx, score, info,
                    Iinv, thresh, stat_type)
                stat, index, tau = _update_res(new_stat, stat, new_index, index,
                    hi, tau)
        else:  # non-Markovian dependency
            for t in trange:
                score_t, info_t = self.information(inputs[:t], targets[:t])
                score_t = score - score_t
                info_t = info - info_t
                new_stat, new_index = _compute_stats(prange, idx, score, info,
                    Iinv, thresh, stat_type)
                stat, index, tau = _update_res(new_stat, stat, new_index, index,
                    t, tau)
        return _return_results(stat, index, tau, stat_type)
