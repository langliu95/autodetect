Text Topic Models
=================

In this section we introduce a text topic model proposed by
`Stratos et al. (2015) <http://www.cs.columbia.edu/~djhsu/papers/count_words.pdf>`_
to detect changes in language level on text data.
This model is a restricted class of hidden Markov models (HMMs) introduced by
`Brown et al. (1992) <https://www.aclweb.org/anthology/J92-4003>`_, and thus called the Brown model.
To be more specific, it is an HMM with discrete emission distribution such that
each row of the emission matrix (rows for hidden states and columns for word types)
must have a single nonzero entry, henceforth it can be used as a word embedding scheme.

Data description
----------------
A dataset containing subtitles of TV shows is provided (downloaded from `TVsubtitles <http://www.tvsubtitles.net/>`_) in this package.
The first two seasons of four TV shows are collected---Friends, Modern Family, the Sopranos,
and Deadwood.
Then these text data are preprocessed using the following steps:
1) replace contractions with their original forms, 2) remove punctuations, 3) tokenization,
4) remove person names, 5) convert to lower case, 6) replace numbers with their words
equivalent, 7) remove stopwords, 8) lemmatization.
To load the data, you need to move to the root directory of this package, then you may, for example, run

.. code-block:: python

    import pandas as pd
    show1 = pd.read_csv('autodetect/data/subtitles/friends_S1.txt', sep='\n', header=None,
        names=['words'])
    show2 = pd.read_csv('autodetect/data/subtitles/deadwood_S1.txt', sep='\n', header=None,
        names=['words'])

Now we have Season 1 of Friends and Season 1 of Deadwood, and they are apparently different.
One remarkable discrepency is the former is all about conversations between good friends,
while the latter is filled of rude words.
An interesting question to ask is whether the autograd-test can capture this
language-level transition.

Change detection in language level
----------------------------------

One possible solution is to model the data with the Brown model, and detect changes
in model parameters.
For this purpose, we firstly combine these two shows together and remove words of
low frequency, then convert the data to the format that can be fed to the Brown model.

.. code-block:: python

    import numpy as np
    def remove_infrequent(text1, text2, times):
        """Combines two pieces of text and removes infrequent words (< times)."""
        text = text1.append(text2, ignore_index=True)
        counts = text.iloc[:, 0].value_counts()
        rare = counts[counts < times].index
        text = text[~text.iloc[:, 0].isin(rare)]
        return text

    text = remove_infrequent(show1, show2, 15)
    n_states = int(np.sqrt(len(text) / 100))  # 100 observations for each transition parameter
    text.iloc[:, 0] = text.iloc[:, 0].astype('category')
    ints = text.iloc[:, 0].cat.codes
    y = ints.values

Next, we initialize the Brown model and fit it using the dataset to obtain the MLE.

.. code-block:: python

    from autodetect import AutogradTopic
    n_cats = y.max() + 1
    model = AutogradTopic(n_cats, n_states)
    model.train(y)

Finally, noting that emission parameters can easily alter because of the
shift of high-frequency words while transition parameters are purely determined
by hidden states which may incorporate more "global" information,
we perform the autograd-test to detect the existence of a change in
transition parameters.

.. code-block:: python

    trange = np.arange(int(len(y)/4), int(len(y)*3/4), 10)
        # only detect change in the middle half sample (avoid ill-conditioned information)
    idx = range(n_cats - n_states, n_cats + n_states * (n_states - 2))
        # exclude n_cats - n_states emission pars
    stat, tau, index = model.compute_stats(y, idx=idx, trange=trange, stat_type='scan')

.. note::

    For the sake of time, we only compute statistics for locations with a gap of 10.
    In simulation studies, we observe that the linear part in the autograd-test
    tends to be liberal (has high false discovery rate) when applying to Brown models
    with high-dimensional parameter space. To avoid suffering this drawback we use
    the scan test here.


API reference
-------------
.. autoclass:: autodetect.AutogradTopic
    :members:
