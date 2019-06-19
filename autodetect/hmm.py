"""Module for autograd-test in hidden Markov model.

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


def _prob_mat_pars(mat):
    """Converts a probability matrix to a tensor of effective parameters.

    The effective parameters are the elements of the :math:`n \\times (n - 1)`
    submatrix, where constraints on the probability matrix are removed.
    The conversion to a 1D tensor is done by row.
    """
    Q = torch.tensor(mat[:, :-1]).float()
    return Q.view(-1)


def _construct_prob_mat(pars, dim=None):
    """Constructs probability matrix based on effective parameters.

    The effective parameters should be attained by reshaping the ``dim`` by
    ``dim - 1`` submatrix as a 1D vector by row.

    Parameters
    ----------
    pars: torch.Tensor, 1D
        Effective parameters.
    dim: int, optional
        Dimension of the probability matrix. Default is None.
    """
    if dim is None: dim = int(len(pars)**0.5) + 1
    subQ = pars.view(dim, -1)
    Q = torch.zeros((dim, dim))
    Q[:, :-1] = subQ
    Q[:, -1] = 1 - torch.sum(subQ, 1)
    return Q


class AutogradHmm(object):
    """A class for autograd-test in hidden Markov model.

    This class only supports hidden Markov models (HMMs) with finite hidden
    states.

    Your models should subclass this class with the method
    ``loglike_emission`` being overridden.
    Then you need to use the method ``setup`` to provide the MLE of model parameters
    under null hypothesis (no change exists).

    You may use the module :class:`pomegranate` to train your model.

    .. note::
        This class treats all model parameters as a single parameter vector to
        compute the derivatives of the log-likelihood function.
        For transition (emission) matrix, its parameter vector is
        obtained by reshaping the first ``n_states * (n_states - 1)``
        submatrix by row.
        For non-discrete emission distribution, its parameter vector is
        obtained by reshaping its parameter matrix by row, whose :math:`k`-th
        row contains parameters of emission distribution given hidden state :math:`k`.

    Parameters
    ----------
    n_states: int
        Number of hidden states.
    init: torch.Tensor, optional
        Initial distribution for hidden states. Default is ``None``, which
        will be set to Uniform distribution.
    """

    def __init__(self, n_states, init=None):
        super(AutogradHmm, self).__init__()
        self._n_states = n_states
        self._init = init or torch.ones(n_states) / n_states
        self._tran = torch.tensor([])
        self._emis = torch.tensor([])
        self._is_discrete = None
        self._forward = torch.tensor([])
        self._const = torch.tensor([])

    def parameters(self):
        """Gets model parameters."""
        if self._tran.requires_grad: yield self._tran
        if self._emis.requires_grad: yield self._emis

    def get_transition(self):
        """Gets transition matrix."""
        return _construct_prob_mat(self._tran, self._n_states)

    def get_emission(self):
        """Gets emission parameters.

        If the emission distribution is discrete, returns the emission matrix;
        otherwise returns a matrix whose :math:`k`-th row contains parameters
        of emission distribution given hidden state :math:`k`.
        """
        if self._is_discrete == "Discrete":
            return _construct_prob_mat(self._emis, self._n_states)
        return self._emis.view(self._n_states, -1)

    def get_normalized_forward(self):
        """Gets normalized forward probabilities (filtering)."""
        return self._forward

    def get_normalizing_constant(self):
        """Gets normalizing constants of forward quantities."""
        return self._const

    def write_transition(self, tran, requires_grad=False):
        """Writes transition parameters of the model.

        Parameters
        ----------
        tran: array-like, shape (n_states, n_states)
            Transition matrix (``tran[i, j]`` is the conditional probability
            :math:`p(j|i)`).
        requires_grad: bool, optional
            Gradient status of the transition parameters (set to ``False`` if
            true values are known and provided). Default is ``False``.
        """

        self._tran = _prob_mat_pars(tran)
        self._tran.requires_grad = requires_grad

    def write_emission(self, emis, discrete, requires_grad=False):
        """Writes emission parameters of the model.

        Parameters
        ----------
        emis: array-like, shape (n_states, *)
            Emission parameters. Its :math:`k`-th row contains parameters of emission distribution given hidden state :math:`k`.
        discrete: bool
            Indicates if the emission distribution is discrete.
        requires_grad: bool, optional
            Gradient status of the emission parameters (set to ``False`` if
            true values are known and provided). Default is ``False``.
        """

        if discrete:
            G = torch.tensor(emis[:, :-1]).float()
            self._emis = G.view(-1)
        else:
            self._emis = torch.tensor(emis).float().view(-1)
        self._emis.requires_grad = requires_grad
        self._is_discrete = discrete

    def setup(self, tran, emis, discrete, true_train=False, true_emis=False):
        """Sets up the model.

        This method constructs the HMM based on the transition parameters and
        emission parameters provided. You should indicate if they are true
        values or estimates (unknown). True values will not be considered as
        model parameters.

        Parameters
        ----------
        tran: array-like, shape (n_states, n_states)
            Transition matrix.
        emis: array-like, shape (n_states, *)
            Emission parameters. If emission distribution is discrete, it
            should be emission matrix; otherwise it is a matrix whose
            :math:`k`-th row contains parameters of emission distribution
            given hidden state :math:`k`.
        discrete: bool
            Indicates if emission distribution is discrete.
        true_train: bool, optional
            Indicates if the transition matrix provided is the true value or
            estimate from data. Default is ``False``.
        true_emis: bool, optional
            Indicates if the emission parameters provided are true values or
            estimates from data. Default is ``False``.
        """
        self.write_transition(tran, not true_train)
        self.write_emission(emis, discrete, not true_emis)

    def gradients(self):
        """Gets gradients of model parameters.

        Returns an 1D ``Tensor`` contains the gradient of parameters.
        """
        grads = []
        tran_grad = self._tran.grad
        emit_grad = self._emis.grad
        if tran_grad is not None: grads.append(tran_grad)
        if emit_grad is not None: grads.append(emit_grad)
        return torch.cat(grads)

    def loglike_emission(self, obs, states):
        """Computes log-likelihood for the emission distribution.

        Should be overridden by all subclasses. Use the method ``get_emission``
        to get emission parameters.

        It should at least support a single observation and associated hidden state.

        Parameters
        ----------
        obs: torch.Tensor
            Observations.
        states: array-like (integer)
            Hidden states associated with ``obs``.
        """
        raise NotImplementedError

    def information_emission(self, obs, states):
        """Computes score and information for the emission distribution.

        Parameters
        ----------
        obs: torch.Tensor
            Observations.
        states: array-like (integer)
            Hidden states associated with ``obs``.
        """
        lpar = self._par_length()
        score = torch.zeros(lpar)
        information = torch.zeros((lpar, lpar))
        emis = self._emis
        if not emis.requires_grad:
            return score, information

        loglike = self.loglike_emission(obs, states)
        grads = grad(loglike, emis, allow_unused=True, create_graph=True)[0]
        # stores gradients and information matrix
        n_states = self._n_states
        start = n_states * (n_states - 1)
        ident = torch.eye(len(emis))
        score[start:] = grads
        for j in range(len(emis)):
            if self._emis.grad is not None: self._emis.grad.data.zero_()
            if j < len(emis) - 1:
                grads.backward(ident[j], retain_graph=True)
            else:
                grads.backward(ident[j])
            information[start + j, start:] = -emis.grad
        return score.detach(), information.detach()

    def filtering(self, obs):
        """Performs normalized forward algorithm (filtering).

        This method implements the normalized forward algorithm for
        observations provided and stores the forward probabilities and
        normalizing constants as model attributes. See the book
        "`Inference in Hidden Markov Models <http://people.bordeaux.inria.fr/pierre.delmoral/hmm-cappe-moulines-ryden.pdf>`_"
        for more details.

        Parameters
        ----------
        obs: torch.Tensor
            Observations.
        """

        n = len(obs)
        N = self._n_states
        Q = self.get_transition().detach()
        c = torch.zeros(n)  # normalizing constant
        forwards = torch.zeros((n, N))  # filtering
        g = torch.tensor([torch.exp(self.loglike_emission(obs[0], state)) for
                          state in range(N)])  # emission probabilities
        # initialization
        c[0] = torch.sum(g * self._init)
        forwards[0, :] = g * self._init / c[0]
        for i in range(1, n):
            g = torch.tensor([torch.exp(self.loglike_emission(obs[i], state))
                              for state in range(N)])
            alpha = forwards[i - 1, :] @ Q * g
            c[i] = torch.sum(alpha)
            forwards[i, :] = alpha / c[i]
        self._forward = forwards
        self._const = c

    def zero_grad(self):
        """Sets gradients of the model to zero."""
        if self._tran.grad is not None:
            self._tran.grad.data.zero_()
        if self._emis.grad is not None:
            self._emis.grad.data.zero_()

    def log_likelihood(self, obs, filtered=False):
        """Computes log-likelihood.

        Parameters
        ----------
        obs: torch.Tensor
            Observations.
        filtered: bool, optional
            Indicates if the method ``filtering`` has been called for ``obs``.
        """
        if not filtered:
            self.filtering(obs)
        return torch.sum(torch.log(self._const))

    def information(self, obs, filtered=False):
        """Computes score and information matrix.

        Parameters
        ----------
        obs: torch.Tensor
            Observations.
        filtered: bool, optional
            Indicates if the method ``filtering`` has been called for ``obs``.
        """
        if not filtered:
            self.filtering(obs)
        tau1, tau2, tau3 = self._recursive_smoother(obs)
        score = torch.sum(tau1, 0)
        info = torch.ger(score, score) + torch.sum(tau2 - tau3, 0)
        return score, info

    def compute_stats(self, obs, alpha=0.05, idx=None, prange=None, trange=None, stat_type='autograd'):
        """Computes test statistics.

        This function performs score-based hypothesis tests to detect the existence of a change in a hidden Markov model as it learns from
        a continuous, possibly evolving, stream of data.
        Three tests are implemented: the linear test, the scan test, and the autograd-test. The
        linear statistic is the maximum score statistic over all possible locations of
        change. The scan statistic is the maximum score statistic over all possible
        locations of change, and over all possible subsets of parameters in which change occurs.

        .. note::
            This method will run the ``filtering`` method for ``obs``.

        Parameters
        ----------
        obs: torch.Tensor
            Observations (can be multi-dimensional).
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

        n = len(obs)
        alpha, idx, prange, trange = _exceptions_handling(n, self._par_length(),
            alpha, 0, idx, prange, trange, stat_type)
        # computes the inverse of information matrix once
        ident = torch.eye(self._par_length())
        self.filtering(obs)
        score, info = self.information(obs, True)
        Iinv = _compute_inv(ident, info)
        # computes thresholds once
        thresh = compute_thresholds(len(idx), max([1, max(trange) - min(trange)]),
            alpha, prange, stat_type)
        # computes test statistic
        stat = torch.zeros(3)
        tau = np.array([0, 0, 0])
        index = [idx, np.arange(max(prange)), idx]
        for lo, hi in zip([0, *trange[:-1]], trange):
            if lo == 0:
                tau1, tau2, tau3 = self._recursive_smoother(obs[:hi])
            else:
                tau1, tau2, tau3 = self._smoother_interval(obs[lo:hi], lo,\
                    hi, tau1, tau2, tau3)
            temp = torch.sum(tau1, 0)
            cond_score = score - temp  # conditional score
            temp = torch.ger(temp, temp) + torch.sum(tau2 - tau3, 0)
            cond_info = info - temp  # conditional information
            new_stat, new_index = _compute_stats(prange, idx, cond_score,
                cond_info, Iinv, thresh, stat_type)
            stat, index, tau = _update_res(new_stat, stat, new_index, index, hi, tau)
        return _return_results(stat, index, tau, stat_type)

    def _par_length(self):
        """Gets length of parameter vector."""
        lpar = 0
        if self._emis.requires_grad:
            lpar += len(self._emis)
        if self._tran.requires_grad:
            lpar += len(self._tran)
        return lpar

    def _emission_dist(self, obs):
        """Computes emission distribution of observations conditioning on each state."""
        n = len(obs)
        n_states = self._n_states
        g = torch.zeros((n, n_states))
        for k in range(n):
            for i in range(n_states):
                g[k, i] = torch.exp(self.loglike_emission(obs[k], i))
        return g.detach()

    def _init_smoother(self, obs, phi):
        """Computes initial smoother.
        """

        n_states = self._n_states
        lpar = self._par_length()
        tau1 = torch.zeros((n_states, lpar))
        tau2 = torch.zeros((n_states, lpar, lpar))
        tau3 = torch.zeros((n_states, lpar, lpar))
        for j in range(n_states):
            tau1[j], tau2[j] = self.information_emission(obs, j)
            tau3[j] = torch.ger(tau1[j], tau1[j]) * phi[j]
            tau1[j] = tau1[j] * phi[j]
            tau2[j] = tau2[j] * phi[j]
        return tau1, tau2, tau3

    def _next_smoother(self, obs, k, tau1, tau2, tau3):
        """Computes next smoother.

        Parameters
        ----------
        obs: Next observation.
        k: Index of next observation.
        tau1, tau2, tau3: current smoother.
        """

        n_states = self._n_states
        lpar = self._par_length()
        g = self._emission_dist(obs.view(1, -1)).view(-1).detach()
        phi = self._forward[k-1]
        c = self._const[k]
        q = self.get_transition().detach()
        # for score
        s1 = torch.zeros((n_states, n_states, lpar))
        new1 = torch.zeros((n_states, lpar))
        # for information
        s2 = torch.zeros((n_states, n_states, lpar, lpar))
        new2 = torch.zeros((n_states, lpar, lpar))
        new3 = torch.zeros((n_states, lpar, lpar))
        # computes addtive parts of the smoothing functional
        for i in range(n_states):
            for j in range(n_states):
                if i == 0:
                    s1[:, j], s2[:, j] = self.information_emission(obs, j)
                if j < n_states - 1:
                    ind = i * (n_states - 1) + j
                    s1[i, j, ind] = 1 / q[i, j]
                    s2[i, j, ind, ind] = 1 / q[i, j]**2
                else:
                    low = i * (n_states - 1)
                    high = (i + 1) * (n_states - 1)
                    s1[i, j, low:high] = -1 / q[i, j]
                    s2[i, j, low:high, low:high] = 1 / q[i, j]**2

        for j in range(n_states):
            new1[j] = (q[:, j] @ tau1 + (phi * q[:, j]) @ s1[:, j]) * g[j] / c
            new2[j] = torch.sum(q[:, j][:, np.newaxis, np.newaxis] * tau2, 0)\
                * g[j] / c
            new2[j] += torch.sum((phi * q[:, j])[:, np.newaxis, np.newaxis]
                                 * s2[:, j], 0) * g[j] / c
            for i in range(n_states):
                sxtau = torch.ger(s1[i, j, :], tau1[i])
                s1xs1 = torch.ger(s1[i, j, :], s1[i, j, :])
                new3[j] += (tau3[i] + sxtau.transpose(0, 1) + sxtau + phi[i]
                            * s1xs1) * q[i, j] * g[j]
            new3[j] /= c
        return new1, new2, new3

    def _smoother_interval(self, obs, lo, hi, tau1, tau2, tau3):
        """Computes smoother within an interval."""
        new1, new2, new3 = tau1, tau2, tau3
        for i, t in enumerate(range(lo, hi)):
            new1, new2, new3 = self._next_smoother(obs[i], t, new1, new2, new3)
        return new1, new2, new3

    def _recursive_smoother(self, obs):
        """Computes recursive smoother associated with score and information.

        Run the method ``filtering`` first.
        """
        n = len(obs)
        n_states = self._n_states
        lpar = self._par_length()
        phi = self._forward[0]
        # for score
        tau1 = torch.zeros((n_states, lpar))
        # for information
        tau2 = torch.zeros((n_states, lpar, lpar))
        tau3 = torch.zeros((n_states, lpar, lpar))
        # initialization
        tau1, tau2, tau3 = self._init_smoother(obs[0], phi)
        for k in range(1, n):
            tau1, tau2, tau3 = self._next_smoother(obs[k], k, tau1, tau2, tau3)
        return tau1, tau2, tau3
