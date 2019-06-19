"""Utils for simulations.

Author: Lang Liu
Date: 06/10/2019
"""

from __future__ import absolute_import, division, print_function

import sys

import numpy as np
import numpy.random as npr
from pomegranate import HiddenMarkovModel
from pomegranate import NormalDistribution
from statsmodels.tsa.arima_model import ARMA
import torch

sys.path.append("..") # Adds higher directory to python modules path.
from autodetect.data import Generator


##########################################################################
# functions for generating parameters
##########################################################################

def prob_matrix(n, m, alpha=None):
    """Generates a probability matrix.

    This function generates a probability matrix whose each row has a Dirichlet
    distribution.

    Parameters
    ----------
    d: int
        Dimension of the matrix.
    alpha: array-like, shape (d,)
        Parameters of the Dirichlet distribution.
    """

    alpha = alpha or np.ones(m)
    prob_mat = np.zeros((n, m))
    for i in range(n):
        temp = np.ones(m) / 2 / m
        temp += npr.dirichlet(alpha) / 2
        temp_max = np.max(temp)
        temp_ind = np.argmax(temp)
        prob_mat[i] = temp
        prob_mat[i, i] = temp_max
        prob_mat[i, temp_ind] = temp[i]
    return prob_mat


def change_for_prob_vec(vec, p):
    """Generates change for probability vector.

    The number of positive changes is equal to the one of negative changes, or
    their difference is one.

    Parameters
    ----------
    vec: array-like, shape (d)
        Probability vector in which the change is generated.
    p: int
        Number of changed components. Must be positive.
    """

    delta = np.zeros(len(vec))
    order = np.argsort(vec[:-1])  # the last element is not considered as parameter.
    pos_num = neg_num = int(p / 2)

    if pos_num > 0:
        delta[order[:pos_num]] = 1.0
        delta[order[-neg_num:]] = -1.0
        maximum = min(1 - vec[order[pos_num-1]], vec[order[-neg_num]])
    else:
        maximum = 1.0
    # decides the last change
    if p > 2 * pos_num:
        if vec[order[-(neg_num+1)]] >= vec[-1]:
            neg_num += 1
            delta[-1] = 1.0
            delta[order[-neg_num]] = -1.0
            maximum = min(maximum, vec[order[-neg_num]], 1 - vec[-1])
        else:
            delta[-1] = -1.0
            delta[order[pos_num]] = 1.0
            maximum = min(maximum, 1 - vec[order[pos_num]], vec[-1])
            pos_num += 1
    return delta, maximum


def change_for_prob_matrix(prob_mat, p):
    """Generates change for probability matrix.

    Components of change are as even as possible across rows. Given a row, the
    number of positive changes is equal to the one of negative changes, or
    their difference is one.

    Parameters
    ----------
    prob_mat: array-like, shape (d, d)
        Probability matrix in which the change is generated.
    p: int
        Number of changed components. Must smaller than :math:`d(d-1)`.
    """

    d, c = prob_mat.shape
    delta = np.zeros((d, c))
    maximum = 1
    quot = int(p / d)  # number of changed components of each row
    for k in range(d):
        if k < p % d:
            num = quot + 1
        else:
            num = quot
        if num > 0:
            delta[k], new_max = change_for_prob_vec(prob_mat[k], num)
            maximum = min(maximum, new_max)
    return delta, maximum


def change_for_brown_keep_zero(prob_mat, p):
    """Generates change for probability matrix in Brown model.

    This function does not impose changes in component that is zero.
    Components of change are as even as possible across rows.

    Parameters
    ----------
    prob_mat: array-like, shape (d, d)
        Probability matrix in which the change is generated.
    p: int
        Number of changed components. Must smaller than :math:`d(d-1)`.
    """

    d, c = prob_mat.shape
    delta = np.zeros((d, c))
    maximum = 1
    quot = int(p / d)  # number of changed components of each row
    for k in range(d):
        if k < p % d:
            num = quot + 1
        else:
            num = quot
        non_zero = np.arange(c)[prob_mat[k] > 0]
        if num >= len(non_zero):
            raise RuntimeError(f"Too many zero entries in {k+1}th row of prob_mat")
        if num > 0:
            delta[k, non_zero], new_max = change_for_prob_vec(prob_mat[k, non_zero], num)
            maximum = min(maximum, new_max)
    return delta, maximum


def pars_for_hmm(n, c, p_tran, p_emis, emission="Normal", alpha=None):
    """Generates parameters for HMM.

    This function generates the transition matrix and emission matrix by
    using Dirichlet distribution.

    Parameters
    ----------
    n: int
        Number of hidden states.
    c: int
        Number of parameters.
    emission: string
        Emission distribution. Default is "Normal".
    alpha: array-like, shape (n,)
        Parameters of the Dirichlet distribution for transition matrix.
    """

    tran = prob_matrix(n, n, alpha)
    delta_tran, max_tran = change_for_prob_matrix(tran, p_tran)

    emis = np.zeros((n, 2))
    delta_emis = np.zeros((n, c))
    if emission == "Normal":
        emis[:, 0] = np.arange(n)
        emis[:, 1] = np.linspace(0.01, 0.1, n)
        delta_emis[range(p_emis), 0] = -1.0
        max_emis = p_emis
    elif emission == "Discrete":
        emis = prob_matrix(n, c, alpha)
        delta_emis, max_emis = change_for_prob_matrix(emis, p_emis)

    return {"transition": [tran, delta_tran, max_tran], "emission": [emis, delta_emis, max_emis]}


def pars_for_brown(n, c, p_tran, p_emis, alpha=None):
    """Generates parameters for Brown model.

    This function generates the transition matrix and emission matrix by
    using Dirichlet distribution.

    Parameters
    ----------
    n: int
        Number of hidden states.
    c: int
        Number of categories of emission distribution.
    p: int
        Number of changed components.
    alpha: array-like, shape (n,)
        Parameters of the Dirichlet distribution for transition matrix.

    Returns
    -------
    A dictionary containing all parameters.
    """

    tran = prob_matrix(n, n, alpha)
    delta_tran, max_tran = change_for_prob_matrix(tran, p_tran)

    emis = np.zeros((n, c))
    min_state = int(p_emis / n) + 1
    resi = p_emis % n
    asign_states = npr.choice(range(n), c)
    lo = 0
    for s in range(n):
        if s < resi:
            num = min_state + 1
        else:
            num = min_state
        asign_states[lo:(lo+num)] = s
        lo = lo + num
    npr.shuffle(asign_states)
    for s in range(n):
        in_state_s = asign_states == s
        num_of_s = np.sum(in_state_s)
        emis[s, in_state_s] = np.ones(num_of_s) * 3 / num_of_s / 4
        emis[s, in_state_s] += npr.dirichlet(np.ones(num_of_s)) / 4

    delta_emis, max_emis = change_for_brown_keep_zero(emis, p_emis)
    return {"transition": [tran, delta_tran, max_tran], "emission": [emis, delta_emis, max_emis]}


def pars_for_arma(p, q, seed):
    """Generates parameters for ARMA model."""
    npr.seed(seed)
    ar_root = np.random.exponential(0.5, p) + 1.0
    ar_root *= np.random.choice([-1, 1], p)
    ma_root = np.random.exponential(0.5, q) + 1.0
    ma_root *= np.random.choice([-1, 1], q)
    phi = np.polynomial.polynomial.polyfromroots(ar_root)
    phi /= -phi[0]
    the = np.polynomial.polynomial.polyfromroots(ma_root)
    the /= the[0]
    return phi, the, ar_root, ma_root


def change_for_ar(phi, ar_root, r):
    """Generates change for AR model."""
    ar_new_root = (1 + r) * ar_root
    phi_new = np.polynomial.polynomial.polyfromroots(ar_new_root)
    phi_new /= -phi_new[0]
    delta = phi_new - phi
    de = np.sqrt(np.sum(delta**2)) / np.sqrt(len(phi))
    return delta, de


def change_for_ma(the, ma_root, r):
    """Generates change for MA model."""
    ma_new_root = ma_root + r * ma_root
    the_new = np.polynomial.polynomial.polyfromroots(ma_new_root)
    the_new /= the_new[0]
    delta = the_new - the
    de = np.sqrt(np.sum(delta**2)) / np.sqrt(len(the))
    return delta, de

##########################################################################
# functions for generating data
##########################################################################

def load_parameters(file_name):
    """Loads parameters for HMM and Brown"""
    with open(file_name) as f:
        pars = f.readlines()

    # transition parameters
    d = int(pars[0])
    tran = np.zeros((d, d))
    delta_tran = np.zeros((d, d))
    loc = 2
    for i in range(d):
        for j in range(d):
            tran[i, j] = float(pars[loc])
            loc += 1
    for i in range(d):
        for j in range(d):
            delta_tran[i, j] = float(pars[loc])
            loc += 1
    max_tran = float(pars[loc])
    loc += 1

    # emission parameters
    p, q = int(pars[loc]), int(pars[loc+1])
    emis = np.zeros((p, q))
    delta_emis = np.zeros((p, q))
    loc += 2
    for i in range(p):
        for j in range(q):
            emis[i, j] = float(pars[loc])
            loc += 1
    for i in range(p):
        for j in range(q):
            delta_emis[i, j] = float(pars[loc])
            loc += 1
    max_emis = float(pars[loc])
    return [tran, delta_tran, max_tran], [emis, delta_emis, max_emis]


def synthetic_data_hmm(n, dim, tau, tran, delta, emis, model, obs_per_state, seed):
    """Generates synthetic data for HMMs.

    model: "Normal" or "Discrete"
    """

    N = len(tran)
    n_cats = emis.shape[1]
    # generates data with change in transition parameters
    nu = np.ones(N) / N
    gen = Generator(n, dim, tau)

    np.random.seed(seed)
    bad = True
    while bad:
        x, y = gen.hmm_transition(tran, delta, emis, model, nu)
        bad = False
        for state in range(N):
            if np.sum(x == state) < obs_per_state:
                bad = True
                break
        if model == 'Discrete':
            for cat in range(n_cats):
                if np.sum(y == cat) < obs_per_state:
                    bad = True
                    break
    return x, y


def synthetic_data_arma(n, dim, tau, phi, delta, the, seed):
    """Generates synthetic data for ARMA models."""
    p, q = len(phi), len(the)
    gen = Generator(n, dim, tau)
    np.random.seed(seed)
    y = gen.arma(phi, delta, the)
    while True:
        try:
            cmle = ARMA(y, order=(p, q)).fit(method='css', trend='nc')
            break
        except ValueError:
            y = gen.arma(phi, delta, the)
            print(y[0])
    obs = torch.from_numpy(np.column_stack([y, np.zeros(n)])).float()
    theta_hat = torch.tensor(cmle.params).float()
    theta_hat.requires_grad = True
    return obs, theta_hat, cmle.sigma2


##########################################################################
# miscellaneous
##########################################################################

def write_mat(matrix, f):
    """Writes matrix to file."""
    for _, mat in enumerate(matrix):
        for v in mat:
            f.write("%s\n" % str(v))


def hmm_mle(y, N):
    """Computes the MLE of normal HMMs."""
    model = HiddenMarkovModel()
    model = model.from_samples(NormalDistribution, N, [y])
    tran = model.dense_transition_matrix()[:N, :N]
    emis = []
    states = model.get_params()['states']
    for s in range(N):
        emis += states[s].distribution.parameters
    emis = np.array(emis).reshape(N, -1)
    return tran, emis


def check_rejection(stat, type_stat=''):
    """Check rejection given the statistic."""
    rej = 0
    if stat > 1: rej = 1
    if stat == 1: print(f"{type_stat}non-invertible rejection.")
    return rej


def check_rejections(stats):
    """Check rejections given three statistics."""
    rej = np.zeros(3, int)
    rej[0] = check_rejection(stats[0], 'Linear ')
    rej[1] = check_rejection(stats[1], 'Scan ')
    rej[2] = check_rejection(stats[2], 'Autograd ')
    print("")
    return rej
