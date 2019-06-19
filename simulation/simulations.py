"""Functions for simulations.

Author: Lang Liu
Date: 06/10/2019
"""

import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

sys.path.append("..") # Adds higher directory to python modules path.
from autodetect import AutogradCuSum, AutogradHmm, AutogradTopic, autograd_arma, autograd_test
from autodetect.data import Generator, Linear, loglike_linear
from utils import (change_for_ar, change_for_ma, check_rejections, hmm_mle,
                   load_parameters, pars_for_arma, synthetic_data_arma,
                   synthetic_data_hmm)


def initialization(model, n, dim, num, dim2=None):
    # print parameters
    print(model)
    print(f"n = {n}")
    if model == 'arma':
        print(f"p = {dim}")
        print(f"q = {dim2}")
    if model == 'brown':
        print(f"N = {dim}")
        print(f"M = {dim2}")
    if model == 'hmm':
        print(f"N = {dim}")
    if model == 'linear':
        print(f"dim = {dim}")

    rej = np.zeros((3, num))
    tau_dist = np.zeros((3, num))
    run_time = np.zeros(num)
    return rej, tau_dist, run_time


def get_file_name(model, n, dim, rep, postfix='', dim2=None):
    if model == 'arma':
        file_name = 'text_files/arma' + str(n) + 'p' + str(dim) + 'q' + str(dim2) + postfix + '_results_' + str(rep) + '.txt'
    elif model == 'brown':
        file_name = 'text_files/brown' + str(n) + 'S' + str(dim) + postfix + '_results_' + str(rep) + '.txt'
    elif model == 'hmm':
        file_name = 'text_files/hmm' + str(n) + 'S' + str(dim) + postfix + '_results_' + str(rep) + '.txt'
    elif model == 'linear':
        file_name = 'text_files/linear' + str(n) + '_dim' + str(dim) + postfix + '_results_' + str(rep) + '.txt'
    return file_name


def run_arma(rep, n, tau0, P, Q, num, seed, postfix=''):
    """Run experiments for ARMA model."""

    # decides parameters
    dim = 1
    #seed = 11919  # for ARMA(3,2)
    #seed = 11519  # for ARMA(6,5)
    lag = max(P, Q)
    phi, the, ar_root, _ = pars_for_arma(P, Q, seed)

    de_range = np.zeros(num)
    root_range = np.linspace(0, 0.25, num)
    rej, tau_dist, run_time = initialization('arma', n, P, num, Q)
    for s, r in enumerate(root_range):
        # decides change
        delta, de_range[s] = change_for_ar(phi, ar_root, r)
        delta = np.r_[delta, np.zeros(Q)]
        # delta, de_range[s] = change_for_ma(the, ma_root, r)
        # delta = np.r_[np.zeros(P), delta]

        # generates data
        y, theta_hat, sig2 = synthetic_data_arma(n, dim, tau0, phi[1:], delta,
                                                 the[1:], s + num * int(rep))

        print('Finished generating data')
        idx = range(P + Q)
        if postfix == '_idx':
            idx = range(P, P + Q)
        start = time.perf_counter()
        stat, tau, _ = autograd_arma(theta_hat, sig2, y, P, Q, idx=idx, stat_type='all')
        end = time.perf_counter()
        run_time[s] = end - start

        print(f'de = {de_range[s]}')
        print(stat, tau)
        tau_dist[:, s] = np.abs(tau - tau0)
        rej[:, s] = check_rejections(stat)
    res = np.row_stack((de_range, rej, tau_dist, run_time))
    file_name = get_file_name('arma', n, P, rep, postfix, Q)
    np.savetxt(file_name, res, delimiter=',')


def run_brown(rep, n, tau0, n_states, num, postfix=''):
    """Run experiments for text topic (Brown) model."""

    # load parameters
    tran_pars, emis_pars = load_parameters('text_files/brownS' + str(n_states) + '_pars.txt')
    tran0, delta_tran, max_tran = tran_pars[0], tran_pars[1], tran_pars[2]
    emis0 = emis_pars[0]
    de_range = np.linspace(0, 0.25, num)

    n_cats = emis0.shape[1]
    rej, tau_dist, run_time = initialization('brown', n, n_states, num, n_cats)
    for s, de in enumerate(de_range):
        # generates data
        delta = de * delta_tran
        seed = s + num * rep
        x, y = synthetic_data_hmm(n, 1, tau0, tran0, delta, emis0,
                                      'Discrete', 80, seed)
        print('Finished generating data')
        # train the model
        model = AutogradTopic(n_cats, n_states)
        model.train(y)

        start = time.perf_counter()
        stat, tau, _ = model.compute_stats(y, stat_type='all')
        end = time.perf_counter()
        run_time[s] = end - start

        print(f'de = {de}')
        print(stat, tau)
        tau_dist[:, s] = np.abs(tau - tau0)
        rej[:, s] = check_rejections(stat)
    res = np.row_stack((de_range, rej, tau_dist, run_time))
    file_name = get_file_name('brown', n, n_states, rep, postfix=postfix)
    np.savetxt(file_name, res, delimiter=',')


class MyHMM(AutogradHmm):
    """HMM with normal emission."""
    def __init__(self, n_states):
        super(MyHMM, self).__init__(n_states)

    def loglike_emission(self, obs, states):
        emis = self.get_emission()[states, :]
        return -((obs - emis[0]) / emis[1])**2 / 2 -\
            torch.tensor(np.log(2 * math.pi)) / 2 - torch.log(emis[1])


def run_hmm(rep, n, tau0, n_states, num, single=False, postfix=''):
    """Run experiments for HMM."""

    # load parameters
    tran_pars, emis_pars = load_parameters('text_files/hmmS' + str(n_states) + '_pars.txt')
    tran0, delta_tran, _ = tran_pars[0], tran_pars[1], tran_pars[2]
    emis0 = emis_pars[0]

    if single:
        print("hmm")
        print(f"n = {n}")
        print(f"N = {len(tran0)}")
        batch = int((rep - 1) / 200)
        print(f"batch = {batch}")
        postfix = '_single'
        de_range = np.linspace(0, 0.25, num)[batch:(batch+1)]
        rej = np.zeros(3, int)
        run_time = np.zeros(num)
        seed = rep
    else:
        de_range = np.linspace(0, 0.25, num)
        rej, tau_dist, run_time = initialization('hmm', n, n_states, num)

    for s, de in enumerate(de_range):
        # generates data
        delta = de * delta_tran
        if not single: seed = s + num * rep
        x, y = synthetic_data_hmm(n, 1, tau0, tran0, delta, emis0,
                                      'Normal', 80, seed)
        print('Finished generating data')
        # computes mle
        tran, emis = hmm_mle(y, n_states)
        # set up the model
        model = MyHMM(n_states)
        model.setup(tran, emis, False)
        inputs = torch.from_numpy(y).float()
        model.filtering(inputs)

        start = time.perf_counter()
        stat, tau, _ = model.compute_stats(inputs, stat_type='all')
        end = time.perf_counter()
        run_time[s] = end - start

        print(f'de = {de}')
        print(stat, tau)
        if single:
            tau_dist = np.abs(tau - tau0)
            rej = check_rejections(stat)
        else:
            tau_dist[:, s] = np.abs(tau - tau0)
            rej[:, s] = check_rejections(stat)
    if single:
        res = np.concatenate((de_range, rej, tau_dist, run_time))
    else:
        res = np.row_stack((de_range, rej, tau_dist, run_time))
    file_name = get_file_name('hmm', n, n_states, rep, postfix=postfix)
    np.savetxt(file_name, res, delimiter=',')


def run_linear(rep, n, tau0, dim, num, postfix=''):
    """Run experiments for linear regression model."""

    # decides parameters
    gen = Generator(n, dim, tau0)
    beta0 = np.zeros(dim + 1)
    # decides changes
    de_range = np.linspace(0, 0.5, num)
    sign = np.zeros(dim + 1)
    sign[0] = 1.0
    if postfix == '_p20':
        sign[0:20] = 1.0

    rej, tau_dist, run_time = initialization('linear', n, dim, num)
    for s, de in enumerate(de_range):
        # generates data
        np.random.seed(s + num * int(rep))
        delta = de * sign
        obs, beta_hat = gen.linear(beta0, delta)
        idx = range(dim)
        if postfix == '_idx':
            idx = range(50)
        if postfix == '_idx_wrong':
            idx = range(50, 100)

        start = time.perf_counter()
        stat, tau, _ = autograd_test(beta_hat, obs, loglike_linear, idx=idx, stat_type='all')
        end = time.perf_counter()
        run_time[s] = end - start

        print(f'de = {de}')
        print(stat, tau)
        tau_dist[:, s] = np.abs(tau - tau0)
        rej[:, s] = check_rejections(stat)
    res = np.row_stack((de_range, rej, tau_dist, run_time))
    file_name = get_file_name('linear', n, dim, rep, postfix=postfix)
    np.savetxt(file_name, res, delimiter=',')



def loglike(out, tar):
    loss_fn = nn.MSELoss(size_average=False)
    return -loss_fn(out, tar) / 2


def run_autocusum(rep, n, tau, train_size, dim, thresh, num):
    """Run experiments for autograd-test-CuSum.
    """
    print(f"n = {n}")
    print(f"tau = {tau}")
    print(f"dim = {dim}")
    print(f"thresh = {thresh}")

    N = n + train_size
    print(f"sample size of training set {train_size}")
    tau = tau + train_size
    de_range = np.linspace(0, 0.5, num)
    sign = torch.zeros(dim)
    sign[0] = 1.0

    rej_length = np.ones(num, int) * n  # initialize with n
    run_time = np.zeros(num)
    for s, de in enumerate(de_range):
        np.random.seed(s + num * rep)
        delta = de * sign
        inputs = torch.randn((N, dim))

        # generates targets with change
        targets = torch.randn(N)
        targets[tau:] += inputs[tau:] @ delta
        #targets = targets.view(-1)

        # pre-trains the model
        linear = Linear(dim, 1)
        optim = torch.optim.Adam(linear.parameters())
        for _ in range(10000):
            optim.zero_grad()
            outs = linear(inputs[:train_size])
            loss = -loglike(outs, targets[:train_size])
            loss.backward()
            optim.step()
        autocusum = AutogradCuSum(linear, loglike)

        # change detection
        start = time.perf_counter()
        autocusum.initial_model(inputs[:train_size], targets[:train_size], N)
        for i in range(train_size, N):
            stat = autocusum.compute_stats(inputs[i:(i+1)], targets[i:(i+1)], thresh)
            if stat > 1.0:
                rej_length[s] = i - tau
                break
        end = time.perf_counter()
        run_time[s] = end - start
        print(f'de = {de}')
        print(stat, rej_length[s])
    res = np.row_stack((de_range, rej_length, run_time))
    file_name = 'text_files/autocusum' + str(dim) + '_' + str(n) + '_' + str(tau) + '_' + str(train_size) + '_results_' + str(rep) + '.txt'
    np.savetxt(file_name, res, delimiter=',')
