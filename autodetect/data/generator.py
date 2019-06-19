"""A class for generating simulation data.

Author: Lang Liu
Date: 06/10/2019
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import numpy.random as npr

import torch


class Generator(object):
    """A class for sampling synthetic data with change in parameters.

    Parameters
    ----------
    size: int
        Sample size.
    dim: int
        Dimension of observations.
    tau: int
        Location of the change (set to be size if no change exists).
    """

    def __init__(self, size, dim, tau):
        self._size = size
        self._dim = dim
        self._tau = tau

    def gaussian(self, mu, delta, sig=1.0):
        """Generates observations from multivariate normal with change in mean.

        Observations are independent and identically distributed, and
        the covariance matrix is a scalar matrix.

        Parameters
        ----------
        mu: array-like, shape (dim,)
            Mean of the model.
        delta: array-like, shape (dim,)
            Value of change in mean. After the change point, the mean of the
            model becomes ``mu + delta``.
        sig: double, optional
            Standard deviation of the model. :math:`sig^2` is the scalar
            of the covariance matrix. Default is 1.0.

        Returns
        -------
        obs: torch.Tensor, shape (size, dim)
            Generated observations.
        mu_hat: torch.Tensor, shape (dim,)
            Maximum likelihood estimate (MLE) of the mean assuming no change.
        """

        x = npr.normal(loc=mu, scale=sig, size=(self._size, self._dim))
        x[self._tau:self._size, :] += delta[np.newaxis, :]
        obs = torch.from_numpy(x).float()
        mu_hat = torch.mean(obs, 0)
        mu_hat.requires_grad_(True)
        return obs, mu_hat

    def linear(self, beta, delta, sig=1.0):
        """Generates observations from linear model with change in coefficients.

        Covariates are sampled from multivariate normal, and the covariance
        matrix is the identity matrix. Errors are sampled from normal distribution.

        Parameters
        ----------
        beta: numpy.ndarray, shape (dim,)
            Coefficients of the linear model.
        delta: numpy.ndarray shape (dim+1,)
            Value of change in coefficients and intercept (the last element).
        sig: double, optional
            Standard deviation of errors. Default is 1.0.

        Returns
        -------
        obs: torch.Tensor, shape (size, dim+2)
            Generated observations. The first ``dim + 1`` columns form the
            design matrix, and the last column contains responses.
        beta_hat: torch.Tensor, shape(dim+1,)
            MLE of the coefficients and intercept assuming no change.
        """

        n = self._size
        d = self._dim
        tau = self._tau
        design = np.column_stack((npr.normal(size=(n, d)), np.ones(n)))
        e = npr.normal(scale=sig, size=n)
        response = design @ beta + e
        response[tau:n] += design[tau:n, :] @ delta
        # compute MLE under H_0
        ident = np.identity(d+1)
        inv = np.linalg.solve(design.T @ design, ident)
        beta_hat = torch.from_numpy(inv @ (design.T @ response)).float()
        beta_hat.requires_grad_(True)
        obs = torch.cat([torch.from_numpy(design),\
            torch.from_numpy(response.reshape(n, 1))], 1).float()
        return obs, beta_hat

    def hmm_transition(self, tran, delta, emit, emission, nu):
        """Generates observations from Hidden Markov model (HMM) with change in transition matrix.

        Observations must be one-dimensional.

        Parameters
        ----------
        tran: numpy.ndarray, shape (n_states, n_states)
            Transition matrix.
        delta: numpy.ndarray, shape (n_states, n_states)
            Value of change in transition matrix.
        emit: array-like, shape (n_states, n_emissions or n_pars)
            Emission parameters.
            For discrete emission distribution, it is the emission matrix;
            for normal distribution, it is the normal parameters.
        emission: str
            Type of emission distribution. Must be ``"Normal"`` or ``"Discrete"``.
        nu: array-like, shape (n_states,)
            Initial distribution.

        Returns
        -------
        states: numpy.ndarray, shape (size,)
            Hidden states associated with observations.
        obs: numpy.ndarray, shape (size,)
            Generated observations.
        """

        n = self._size
        tau = self._tau
        if self._dim != 1:
            raise ValueError('Only one dimensional HMMs are supported')
        # generates states
        states = np.zeros(n, int)
        states[0] = npr.multinomial(1, nu).argmax()
        for k in range(n-1):
            if k < tau - 1:
                states[k+1] = npr.multinomial(1, tran[states[k], :]).argmax()
            else:
                states[k+1] = npr.multinomial(1, tran[states[k], :] +\
                    delta[states[k], :]).argmax()
        # generates observations
        if emission == "Normal":
            obs = np.zeros(n)
            for k in range(n):
                obs[k] = npr.normal(emit[states[k], 0], emit[states[k], 1])
        elif emission == "Discrete":
            obs = np.zeros(n, int)
            emit_num = emit.shape[1]
            for k in range(n):
                obs[k] = npr.choice(emit_num, size=1, p=emit[states[k], :])
        else:
            raise ValueError("Only 'Normal' and 'Discrete' distributions\
                                 are supported")
        return states, obs

    def hmm_emission(self, tran, delta, emit, emission, nu):
        """Generates observations from Hidden Markov model (HMM) with change in emission parameters.

        Observations must be one-dimensional.

        Parameters
        ----------
        tran: array-like, shape (n_states, n_states)
            Transition matrix.
        delta: numpy.ndarray, shape (n_states, n_states)
            Value of change in emission parameters.
        emit: numpy.ndarray, shape (n_states, n_emissions or n_pars)
            Emission parameters.
            For discrete emission distribution, it is the emission matrix;
            for normal distribution, it is the normal parameters.
        emission: str
            Type of emission distribution. Must be ``"Normal"`` or ``"Discrete"``.
        nu: array-like, shape (n_states,)
            Initial distribution.

        Returns
        -------
        states: numpy.ndarray, shape (size,)
            Hidden states associated with observations.
        obs: numpy.ndarray, shape (size,)
            Generated observations.
        """

        n = self._size
        tau = self._tau
        if self._dim != 1:
            raise ValueError('Only one dimensional HMMs are supported')
        # generates states
        states = np.zeros(n, int)
        states[0] = npr.multinomial(1, nu).argmax()
        for k in range(n-1):
            states[k+1] = npr.multinomial(1, tran[states[k], :]).argmax()
        # generates observations
        if emission == "Normal":
            obs = np.zeros(n)
            for k in range(n):
                if k < tau:
                    obs[k] = npr.normal(emit[states[k], 0], emit[states[k], 1])
                else:
                    obs[k] = npr.normal((emit + delta)[states[k], 0],
                                        (emit + delta)[states[k], 1])
        elif emission == "Discrete":
            obs = np.zeros(n, int)
            emit_num = emit.shape[1]
            for k in range(n):
                if k < tau:
                    obs[k] = npr.choice(emit_num, size=1, p=emit[states[k], :])
                else:
                    obs[k] = npr.choice(emit_num, size=1,
                                        p=(emit+delta)[states[k], :])
        else:
            raise ValueError("Only 'Normal' and 'Discrete' distributions are supported")
        return states, obs

    def arma(self, phi, delta, theta):
        """Generates observations from ARMA with change in AR parameters.

        Observations must be one-dimensional.

        Parameters
        ----------
        phi: numpy.ndarray, shape (p,)
            AR parameters.
        delta: numpy.ndarray, shape (p,)
            Value of change in emission parameters.
        theta: numpy.ndarray, shape (q,)

        Returns
        -------
        obs: numpy.ndarray, shape (size,)
            Generated observations.
        """
        p = len(phi)
        if self._dim != 1:
            raise ValueError('Only one dimensional ARMAs are supported')
        e = npr.normal(scale=0.1, size=self._size)
        obs = np.zeros(self._size)
        obs[0] = e[0]
        for t in np.arange(1, self._size):
            pm = min([t, p])
            qm = min([t, len(theta)])
            yflip = np.fliplr([obs[(t-pm):t]])[0]
            eflip = np.fliplr([e[(t-qm):t]])[0]
            #print(eflip.shape)
            #print(delta[p:(p+qm)].shape)
            #print(theta[0:qm].shape)
            if t < self._tau:
                obs[t] = np.sum(phi[0:pm] * yflip) + e[t] +\
                    np.sum(theta[0:qm] * eflip)
            else:
                obs[t] = np.sum((phi[0:pm] + delta[0:pm]) * yflip) + e[t] +\
                    np.sum((theta[0:qm] + delta[p:(p+qm)]) * eflip)
        return obs
