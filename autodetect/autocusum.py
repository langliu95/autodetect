"""Module for autograd-test-CuSum (online variant of autograd-test).

Author: Lang Liu
Date: 06/10/2019
"""

from __future__ import absolute_import, division, print_function

import copy
import numpy as np

import torch

from .utils import _compute_culinear_stat


class AutogradCuSum(object):
    """A class for change point detection via autograd-test-CuSum.

    This is an online variant of autograd-test, and it only supports independent data.

    .. note::
        This class treats all model parameters as a single parameter vector to
        compute the derivatives of the log-likelihood function.
        This parameter vector is obtained by iterating over parameters of your
        pre-trained model and reshaping each of them to a vector by row.

        The log-likelihood function is closely related to loss function in
        machine learning literature. Negative log-likelihood functions can be
        used as loss functions; while some loss functions have corresponding
        log-likelihood functions (such as mean square error versus
        log-likelihood of Gaussian).

    Parameters
    ----------
    pretrained_model: torch.nn.Module
        A pre-trained model inherited from :class:`torch.nn.Module`.
    loglike: function
        ``loglike(outputs, targets)`` is the log-likelihood of model parameters
        given ``outputs`` and ``targets``.
    """

    def __init__(self, pretrained_model, loglike):
        super(AutogradCuSum, self).__init__()
        self._dim = sum(par.numel() for par in pretrained_model.parameters() if
                        par.requires_grad)
        self._model = copy.deepcopy(pretrained_model)
        self._loglike = loglike
        self._initialization()

    def model_parameters(self):
        """Get model parameters.
        """
        pars = []
        for par in self._model.parameters():
            par = par.view(-1)
            pars.append(par)
        return torch.cat(pars).detach()

    def log_likelihood(self, inputs, targets):
        """Compute log-likelihood."""
        outputs = self._model(inputs)
        return self._loglike(outputs, targets)

    def gradients(self):
        """Get gradients of model parameters.

        Returns a 1D ``Tensor`` contains the derivatives of each parameters in
        the parameter vector.
        """
        grads = []
        for par in self._model.parameters():
            grads.append(par.grad.view(-1))  # removed grad[0]
        return torch.cat(grads)

    def _initialization(self):
        self._size = 0
        self._max_size = 0
        self._rest_size = 0
        self._score = torch.zeros(self._dim)
        self._info = torch.zeros((self._dim, self._dim))
        self._full_score = torch.zeros(self._dim)
        self._full_info = torch.zeros((self._dim, self._dim))
        self._min_stat = -1.0

    def initial_model(self, inputs, targets, max_size):
        """Set up inital model.

        Inputs and targets are the data used for training, and they must be
        iterable.
        """
        self._initialization()
        self._rest_size = self._max_size = max_size
        for x, y in zip(inputs, targets):
            self._model.zero_grad()
            like = self.log_likelihood(x, y)
            like.backward()
            temp = self.gradients().detach()
            self._score += temp
            self._info += torch.outer(temp, temp)
            self._full_score += temp
            self._full_info += torch.outer(temp, temp)
            self._size += 1

    def _update_model(self, inputs, targets):
        """Update model given new (inputs, targets).

        This method takes one step update of Adam using new (inputs, targets).
        """
        optim = torch.optim.Adam(self._model.parameters())
        optim.zero_grad()
        loss = -self.log_likelihood(inputs, targets)
        loss.backward()
        optim.step()

    def update_information(self, inputs, targets):
        """Compute score function and information matrix (first derivatives).

        .. note::
            This function will set gradients of the model to zero.

        Parameters
        ----------
        inputs: torch.Tensor, shape (size, dim)
        targets: torch.Tensor, shape (size, *)

        """
        # do no update existing score and info
        for x, y in zip(inputs, targets):
            # updates model
            self._update_model(x, y)
            self._model.zero_grad()
            # computes score
            like = self.log_likelihood(x, y)
            like.backward()
            temp = self.gradients().detach()
            # updates statistics
            self._score += temp
            self._info += torch.outer(temp, temp)
            self._full_score += temp
            self._full_info += torch.outer(temp, temp)
            self._size += 1

    def compute_stats(self, inputs, targets, thresh, min_thresh=0.0, idx=None):
        """Compute statistics given new (inputs, targets).

        Currently only the linear statistic is supported.

        Parameters
        ----------
        inputs: torch.Tensor, shape (size, dim)
        targets: torch.Tensor, shape (size, *)
        thresh: float
            Threshold for the stopping criterion. Use the method ``quantile_max_square_Bessel`` to compute it.
        min_thresh: float, optional
            Reinitialize the process whenever the current statistic is
            below this threshold. Default is zero.
        idx: array-like, optional
            Indices of parameters of interest (the rest parameters are considered constants)
            in the parameter vector.
            Default is ``None``, which will be set to ``range(dim)``.
        """
        if idx is None:
            idx = range(self._dim)
        self.update_information(inputs, targets)

        stat = _compute_culinear_stat(self._full_score[idx],
                                      self._full_info[np.ix_(idx, idx)],
                                      thresh)
        if self._min_stat >= min_thresh:
            if stat <= self._min_stat:
                self._min_stat = max(stat, min_thresh)
                self._score = torch.zeros(self._dim)
                self._info = torch.zeros((self._dim, self._dim))
                #self._rest_size -= self._size
                #self._size = 0
                return 0.0
        else:
            self._min_stat = max(stat, min_thresh)

        stat = _compute_culinear_stat(self._score[idx],
                                      self._info[np.ix_(idx, idx)],
                                      thresh)
        return stat * self._size / self._rest_size
