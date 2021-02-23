"""Module for autograd-test in text topic (Brown) model.

Author: Lang Liu
Date: 06/10/2019
"""

from __future__ import absolute_import, division, print_function

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from .utils import compute_thresholds
from .utils import _compute_stats
from .utils import _exceptions_handling
from .utils import _return_results
from .utils import _update_res


def _count_freq(seq, num):
    """Count frequencies and pairs.

    Parameters
    ----------
    seq: numpy.ndarray, 1D
        A sequence of categorical variables.
    num: int
        Number of categories.
    """

    countz = np.zeros(num, int)
    count_pair = np.zeros((num, num), int)
    # count seq
    unique, counts = np.unique(seq, return_counts=True)
    countz[unique] = counts
    # count pairs
    unique, counts = np.unique(np.c_[seq[:-1], seq[1:]], return_counts=True, axis=0)
    count_pair[unique[:, 0], unique[:, 1]] = counts
    return countz, count_pair


class AutogradTopic():
    """A class for autograd-test in text topic (Brown) model.

    This model can be used to learn word embeddings from text data.
    See the paper "`Model-based Word Embeddings from Decomposition of Count
    Matrices <http://www.cs.columbia.edu/~djhsu/papers/count_words.pdf>`_".

    Parameters
    ----------
    num_cat: int
        Number of categories of the emission distribution.
    num_hid: int
        Number of hidden states.
    init: numpy.ndarray, shape (num_hid,), optional
        Initial distribution. Default is None.

    Attributes
    ----------
    embed: numpy.ndarray, shape (num_cat,)
        Embedding scheme from categories to hidden states.
    """

    def __init__(self, num_cat, num_hid, init=None):
        super(AutogradTopic, self).__init__()
        self._init = init
        self._emis = np.zeros(num_cat - num_hid)  # effective parameters of the emission matrix
        self._cats = np.zeros(num_cat - num_hid, int)  # corresponding category of _emis
        self._tran = np.zeros(num_hid * (num_hid - 1))
        self.embed = np.zeros(num_cat, int)
        self.num_cat = num_cat
        self.num_hid = num_hid

    def get_transition(self):
        """Get transition matrix."""
        subQ = self._tran.reshape(self.num_hid, -1)
        Q = np.zeros((self.num_hid, self.num_hid))
        Q[:, :-1] = subQ
        Q[:, -1] = 1.0 - np.sum(subQ, 1)
        return Q

    def get_emission(self):
        """Get emission matrix."""
        num_cat = len(self.embed)
        num_hid = np.max(self.embed) + 1
        G = np.zeros((num_hid, num_cat))
        pos = 0
        for k in range(num_cat):
            if k in self._cats:
                G[self.embed[k], k] = self._emis[pos]
                pos += 1
            else:
                G[self.embed[k], k] = 1.0 - np.sum(G[self.embed[k], :])
        return G

    def write_transition(self, tran):
        """Write transition matrix."""
        Q = tran[:, :-1]
        self._tran = Q.reshape(-1)

    def write_emission(self, emis):
        """Write emission matrix.

        Also updates the embedding scheme from categories to hidden states.
        """
        # constructs embedding scheme.
        embed = np.arange(self.num_cat)
        states, obs = np.where(emis != 0)
        if len(states) != self.num_cat:
            raise ValueError('The emission matrix does not satisfy the\
                             constraint for the topic model')
        embed[obs] = states
        self.embed = embed
        # write emission parameters and corresponding categories of them
        pos = self.num_cat - self.num_hid - 1  # current position of self_emis
        self_emis = np.zeros(pos + 1)
        self_cats = np.arange(pos + 1)
        # exclude the last non-zero entry for each column in emission matrix.
        exclude = np.zeros(self.num_hid)
        for k in range(self.num_cat - 1, -1, -1):
            if exclude[embed[k]] != 0:
                self_emis[pos] = emis[embed[k], k]
                self_cats[pos] = k
                pos -= 1
            else:
                exclude[embed[k]] = 1
        self._emis = self_emis
        self._cats = self_cats

    def spectral_method(self, obs, transform=np.sqrt, alpha=0.75):
        """Spectral embedding method.

        Parameters
        ----------
        obs: array-like, 1D
            Integer sequence of observations taking values from 0 to
            ``num_cat - 1``.
        transform: function, optional
            Transformation method performed on the frequency of pairs of
            adjacent observations and the frequency of observations. Default
            is ``np.sqrt``.
        alpha: double, optional
            Smoothing parameter. Default is 0.75.

        Returns
        -------
        embed: numpy.ndarray, 1D
            Embedding scheme from categories to hidden states.
        """

        # counts data.
        count = np.zeros((self.num_cat, self.num_cat), int)  # count pairs
        for w in np.arange(len(obs)):
            if w != len(obs) - 1:
                count[obs[w], obs[w + 1]] += 1
            if w != 0:
                count[obs[w], obs[w - 1]] += 1
        num_of_content = transform(count.sum(1))
        num_of_context = transform(count.sum(0))
        count = transform(count)

        # constructs matrix :math:`\\Omega`.
        c_alpha = num_of_context ** alpha
        omega = count * np.sqrt(c_alpha.sum() / num_of_context.sum())\
            / np.sqrt(num_of_content[:, np.newaxis] * c_alpha[np.newaxis, :])
        # SVD.
        u, _, _ = np.linalg.svd(omega, full_matrices=False)
        U = u[:, np.arange(self.num_hid)]
        E = U / np.sqrt(np.sum(U**2, 1)[:, np.newaxis])
        # agglomerative clustering
        clustering = AgglomerativeClustering(n_clusters=self.num_hid).fit(E)
        embed = clustering.labels_
        return embed

    def mle(self, obs, embed, interpolation=False):
        """Compute maximum likelihood estimator assuming no change exists.

        Parameters
        ----------
        obs: array-like, 1D
            Integer sequence of observations taking values from 0 to
            ``num_cat - 1``.
        embed: array-like, 1D
            Embedding scheme from categories to hidden states.

        Returns
        -------
        G: numpy.ndarray, shape (num_hid, num_cat)
            Emission matrix.
        Q: numpy.ndarray, shape (num_hid, num_hid)
            Transition matrix.
        """

        M = self.num_cat
        N = self.num_hid
        G = np.zeros((N, M))  # emission matrix
        Q = np.zeros((N, N))  # transition matrix
        county, _ = _count_freq(obs, M)  # observations
        countx, countq = _count_freq(embed[obs], N)  # states and state pairs
        # compute G and Q
        for w in range(M):
            G[embed[w], w] = county[w] / countx[embed[w]]
        for i in range(N):
            for j in range(N):
                if interpolation:
                    Q[i, j] = 0.9 * countq[i, j] / (countx[i] - (embed[obs[-1]] == i)) + 0.1 * countx[j] / len(obs)
                else:
                    Q[i, j] = countq[i, j] / (countx[i] - (embed[obs[-1]] == i))
        return G, Q

    def train(self, obs, transform=np.sqrt, alpha=0.75, interpolation=False):
        """Train the model.

        Parameters
        ----------
        obs: array-like, 1D
            Integer sequence of observations taking values from 0 to
            ``num_cat - 1``.
        transform: function, optional
            Transformation method performed on the frequency of pairs of
            adjacent observations and the frequency of observations. Default
            is ``np.sqrt``.
        alpha: double, optional
            Smoothing parameter. Default is 0.75.
        """

        embed = self.spectral_method(obs, transform, alpha)
        G, Q = self.mle(obs, embed, interpolation)
        self.write_transition(Q)
        self.write_emission(G)

    def cond_loglike(self, obs):
        """Compute conditional (on ``obs[0]``) log-likelihood."""
        G = self.get_emission()
        Q = self.get_transition()
        states = self.embed[obs]
        cond_loglike = np.sum(np.log(Q[states[:-1], states[1:]])) + \
            np.sum(np.log(G[states[1:], obs[1:]]))
        return cond_loglike, states

    def log_like(self, obs):
        """Compute log-likelihood."""
        nu = self._init
        G = self.get_emission()
        if nu is None:
            raise ValueError("No initial distribution is given")
        else:
            cond_loglike, states = self.cond_loglike(obs)
            return np.log(nu[states[0]]) + np.log(G[states[0], obs[0]]) + cond_loglike(obs)

    def information(self, obs, freq_cats=None, freq_pairs=None):
        """Compute score and information.

        Parameters
        ----------
        obs: array-like, 1D
            Integer sequence of observations taking values from 0 to
            ``num_cat - 1``.
        freq_cats: array-like, shape (num_cat,), optional
            Category frequency in observations. Default is None.
        freq_pairs: array-like, shape (num_hid, num_hid), optional
            Frequency of pairs of adjacent hidden states.
        """

        M = self.num_cat
        N = self.num_hid
        embed = self.embed
        cats = self._cats
        G = self.get_emission()
        Q = self.get_transition()

        if freq_cats is None:
            freq_cats, _ = _count_freq(obs, M)
        if freq_pairs is None:
            _, freq_pairs = _count_freq(embed[obs], N)
        # construct the last word types in each column of emission matrix.
        last_word = np.setdiff1d(np.arange(M), cats)
        last_hid2word = np.arange(N)
        last_hid2word[embed[last_word]] = last_word

        dim = M - N + N * (N - 1)
        score = np.zeros(dim)
        info = np.zeros((dim, dim))
        for d, w in enumerate(cats):
            last = last_hid2word[embed[w]]  # last category with state ``embed[w]``
            score[d] = freq_cats[w] / G[embed[w], w] - freq_cats[last] / G[embed[w], last]
            catsembedw = np.arange(M - N)[embed[cats] == embed[w]] # categories with state embed[w]
            info[d, catsembedw] = freq_cats[last] / G[embed[w], last]**2
            info[catsembedw, d] = freq_cats[last] / G[embed[w], last]**2
            info[d, d] += freq_cats[w] / G[embed[w], w]**2
        for i in range(N):
            ind = M - N + np.arange(i * (N - 1), (i + 1) * (N - 1))
            score[ind] = freq_pairs[i, :-1] / Q[i, :-1] - freq_pairs[i, N - 1] / Q[i, N - 1]
            info[np.ix_(ind, ind)] = freq_pairs[i, N - 1] / Q[i, N - 1]**2
            info[ind, ind] += freq_pairs[i, :-1] / Q[i, :-1]**2
        return score, info

    def compute_stats(self, obs, alpha=0.05, idx=None, prange=None, trange=None, stat_type='autograd'):
        """Compute test statistics.

        This function performs score-based hypothesis tests to detect the existence of a change in a text topic (Brown) model as it learns from
        a continuous, possibly evolving, stream of data.
        Three tests are implemented: the linear test, the scan test, and the autograd-test. The
        linear statistic is the maximum score statistic over all possible locations of
        change. The scan statistic is the maximum score statistic over all possible
        locations of change, and over all possible subsets of parameters in which change occurs.

        Parameters
        ----------
        obs: torch.Tensor, shape (size, dim)
            Observations.
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

        embed = self.embed
        dim = self.num_cat + self.num_hid * (self.num_hid - 2)
        alpha, idx, prange, trange = _exceptions_handling(len(obs), dim, alpha,
                                                          0, idx, prange,
                                                          trange, stat_type)
        # compute the inverse of information matrix once
        county, _ = _count_freq(obs, self.num_cat)
        _, countq = _count_freq(embed[obs], self.num_hid)
        ident = np.identity(dim)
        _, info = self.information(obs, county, countq)
        const = 0
        while True:
            try:
                Iinv = np.linalg.solve(info + const * ident, ident)
                break
            except np.linalg.LinAlgError:
                const += 0.1
                if np.isnan(info).any():
                    print("NAN while computing the information matrix.")
                    print("Please increase the sample size.")
                    return 0
        # compute thresholds once
        thresh = compute_thresholds(len(idx),
                                    max([1, max(trange) - min(trange)]),
                                    alpha, prange, stat_type)
        # compute test statistic
        seg = np.concatenate((np.array([0]), np.array(trange) + 1))  # two consecutive changepoints
        stat = np.zeros(3)
        tau = np.array([0, 0, 0])
        index = [idx, np.arange(max(prange)), idx]
        for k, t in enumerate(trange):
            for s in range(seg[k], seg[k + 1]):
                county[obs[s]] -= 1
                if s > 0: countq[embed[obs[s - 1]], embed[obs[s]]] -= 1
            score, info = self.information(obs[trange[k]:], county, countq)
            new_stat, new_index = _compute_stats(prange, idx, score, info,
                                                 Iinv, thresh, stat_type)
            stat, index, tau = _update_res(new_stat, stat, new_index, index,
                                           t, tau)
        return _return_results(stat, index, tau, stat_type)
