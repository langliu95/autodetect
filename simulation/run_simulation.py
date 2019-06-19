"""Run simulations.

Author: Lang Liu
Date: 06/10/2019
"""

import argparse

from simulations import run_arma
from simulations import run_autocusum
from simulations import run_brown
from simulations import run_hmm
from simulations import run_linear


parser = argparse.ArgumentParser(description='Run simulation for change point detection using gradient-based tests.')
parser.add_argument('model', help='Underlying model for the data')
parser.add_argument('size', type=int, help='sample size')
parser.add_argument('tau', type=int, help='location of the changepoint.')
parser.add_argument('dim', type=int, help='dimension of the parameter space (number of hidden states for HMMs; number of AR parameters for ARMA models)')
parser.add_argument('rep', type=int, help='current repetition number of the experiments')
parser.add_argument('--dim_ma', type=int, help='number of MA parameters for ARMA models')
parser.add_argument('--seed', type=int, help='random seed for generating ARMA parameters')
parser.add_argument('--single', action='store_true', help='whether each experiment is run for a single change value (if True, the repetitions from 200r + 1 to 200(r+1) account for the r-th value)')
parser.add_argument('--train_size', type=int, help='size of the training sample')
parser.add_argument('--thresh', type=float, help='threshold for autograd-test-CuSum')
parser.add_argument('--postfix', default='', help='postfix indicating experimental settings')
parser.add_argument('--num_change', default=20, type=int, help='number of different change values')

args = parser.parse_args()

if args.model == "linear":
    run_linear(args.rep, args.size, args.tau, args.dim, args.num_change, postfix=args.postfix)
elif args.model == "hmm":
    run_hmm(args.rep, args.size, args.tau, args.dim, args.num_change, args.single)
elif args.model == "brown":
    run_brown(args.rep, args.size, args.tau, args.dim, args.num_change)
elif args.model == "arma":
    run_arma(args.rep, args.size, args.tau, args.dim, args.dim_ma, args.num_change, args.seed, postfix=args.postfix)
elif args.model == 'autocusum':
    run_autocusum(args.rep, args.size, args.tau, args.train_size, args.dim, args.thresh, args.num_change)
else:
    print("Only support linear, hmm, brown, arma, and autocusum.")
