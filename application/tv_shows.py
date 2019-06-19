"""Change detection in text data.

Author: Lang Liu
Date: 06/10/2019
"""

import argparse
import sys
import time

import numpy as np
import pandas as pd

sys.path.append("..") # Adds higher directory to python modules path.
from autodetect import AutogradTopic


# define constants
NAME = ['friends_S1', 'friends_S2', 'modern_S1', 'modern_S2',\
     'the_sopranos_S1', 'the_sopranos_S2', 'deadwood_S1', 'deadwood_S2']
NUM = [24, 24, 24, 24, 13, 13, 12, 13]  # number of episodes

parser = argparse.ArgumentParser(description='Changepoint detection on subtitles of tv shows.')
parser.add_argument('num', type=int)
parser.add_argument('--freq', type=int, default=15, help='threshold for removing infrequent words')
parser.add_argument('--obs_per_state', type=int, default=100)
parser.add_argument('--shuffle', action='store_true')
args = parser.parse_args()

# load datasets
def shuffle_episodes(eps, seed):
    np.random.seed(seed)
    np.random.shuffle(eps)


def load_text(i, j, seed=None):
    path = 'processed_subtitles/'  # change path to the directory containing the dataset

    if seed is None:
        path = '../autodetect/data/subtitles/'
        show1 = pd.read_csv(path + NAME[i] + '.txt', sep='\n', header=None, names=['words'])
        show2 = pd.read_csv(path + NAME[j] + '.txt', sep='\n', header=None, names=['words'])
    else:
        show1 = []
        for e in range(NUM[i]):
            show1.append(pd.read_csv(path + NAME[i] + 'E' + str(e+1) + '.txt',
                                     sep='\n', header=None, names=['words']))
        shuffle_episodes(show1, seed)
        show1 = pd.concat(show1, ignore_index=True)

        show2 = []
        for e in range(NUM[j]):
            show2.append(pd.read_csv(path + NAME[j] + 'E' + str(e+1) + '.txt',
                                     sep='\n', header=None, names=['words']))
        shuffle_episodes(show2, seed)
        show2 = pd.concat(show2, ignore_index=True)
    return show1, show2


def remove_infrequent(text1, text2, times):
    """Combines two pieces of text and remove infrequent words (< times)"""

    text = text1.append(text2, ignore_index=True)
    counts = text.iloc[:, 0].value_counts()
    rare = counts[counts < times].index
    # compute the length of text1 after removal
    n1 = len(text1)
    n1 = n1 - text.words[:n1].isin(rare).sum()
    text = text[~text.iloc[:, 0].isin(rare)]
    return text, n1


def change_detection(i, j, N=None, obs_per_state=100, seed=None):
    show1, show2 = load_text(i, j, seed)
    np.random.seed(i * len(NAME) + j)
    text, n1 = remove_infrequent(show1, show2, args.freq)
    if N is None: N = int(np.sqrt(len(text) / obs_per_state))
    dchange = N*(N-1)
    text.iloc[:, 0] = text.iloc[:, 0].astype('category')
    #cats = text.iloc[:, 0].cat.categories
    ints = text.iloc[:, 0].cat.codes
    y = ints.values
    M = y.max() + 1
    # embedding
    model = AutogradTopic(M, N)
    model.train(y, interpolation=False)
    # detection
    trange = np.arange(np.min([int(len(y)/4), n1]), np.max([int(len(y)/4*3), n1]), 10)
    prange = np.arange(1, N)
    idx = range(M-N, M-N+dchange)
    res = model.compute_stats(y, idx=idx, prange=prange, trange=trange, stat_type='scan')
    print(res)
    return res, (N, M, n1)


if __name__ == '__main__':
    l = len(NAME)
    num = args.num
    if args.shuffle:
        print(f"repetition {num}")
        stat = np.zeros((l, l))
        tau = np.zeros((l, l))
        size1 = np.zeros((l, l))
        for i in range(l):
            print(f'the first show is {NAME[i]}')
            for j in range(l):
                print(f'the second show is {NAME[j]}')
                while True:
                    res, pars = change_detection(i, j, obs_per_state=args.obs_per_state, seed=num)
                    if res != 0:
                        break
                    num += 200
                stat[i, j] = res[0]
                tau[i, j] = res[1]
                size1[i, j] = pars[2]
        filename = "tv_shuffle_freq" + str(args.freq) + '_obs' + str(args.obs_per_state)\
            + '_' + str(num) + ".txt"
        np.savetxt(filename, np.row_stack([stat, tau, size1]), delimiter=',')
    else:
        stat, tau, n1 = 0.0, 0, 0
        i = int(num / l)
        j = num % l
        print(f"pair {NAME[i], NAME[j]}")
        res, pars = change_detection(i, j, obs_per_state=args.obs_per_state)
        filename = "tv_shows_freq" + str(args.freq) + '_obs' + str(args.obs_per_state)\
            + '_' + str(num) + ".txt"
        np.savetxt(filename, np.array([res[0], res[1], pars[2]]), delimiter=',')
