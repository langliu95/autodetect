Latent Variable Models
======================

A latent variable model refers to a statistical model that relates
a set of observable variables to a set of latent variables.
It is assumed that responses on observable variables are the
result of an individual's state on the latent variable(s),
and that observable variables are conditionally independent given latent (hidden) variables.
Two well-known examples are `mixture model <https://en.wikipedia.org/wiki/Mixture_model>`_ and `hidden Markov model (HMM) <https://en.wikipedia.org/wiki/Hidden_Markov_model>`_.
For this kind of models, the log-likelihood function is the summation of
full log-likelihood over latent variables.
This indicates that the computation of log-likelihood as well as its first and second
derivatives can be expensive when the latent state space is large.
Hence, specific algorithms are required to compute the test statistic
in a reasonable amount of time.

For HMMs, you can use the class :class:`AutogradHmm` to detect hidden changes.
This class employs the normalized forward
algorithm and the fixed point smoothing technique described in the book
"`Inference in Hidden Markov Models <http://people.bordeaux.inria.fr/pierre.delmoral/hmm-cappe-moulines-ryden.pdf>`_"
to calculate the score function and observed information matrix.

This package does not contain the implementation of the mixture model,
but you may use :class:`AutogradHmm` with a degenerate transition matrix.

Hidden Markov models
--------------------

Here is an example on how to apply this class to detect changepoint in an HMM.
To begin with, let's generate some observations from an HMM with normal emission distribution and with a change in transition matrix:

.. code-block:: python

    import numpy as np
    from autodetect.data import Generator
    n, N, d, tau0 = 1000, 3, 1, 500
    nu = np.array([0.6, 0.1, 0.3])
    tran0 = np.array([[0.55, 0.25, 0.2], [0.2, 0.35, 0.45], [0.5, 0.2, 0.3]])
    delta0 = np.array([[-0.2, 0.1, 0.1], [0.0, 0.0, 0.0], [-0.3, 0.3, 0.0]])
    emis0 = np.array([[-1, 0.1], [0, 0.1], [1, 0.1]])
    gen = Generator(n, d, tau0)
    x, y = gen.hmm_transition(tran0, delta0, emis0, 'Normal', nu)

Next, we proceed by fitting the model to obtain the MLE of transition
and emission parameters under null hypothesis using the package
:class:`pomegranate`.

.. code-block:: python

    from pomegranate import HiddenMarkovModel
    from pomegranate import NormalDistribution
    model = HiddenMarkovModel()
    model = model.from_samples(NormalDistribution, N, [y])
    tran = model.dense_transition_matrix()[:N, :N]
    emis = []
    states = model.get_params()['states']
    for s in range(N):
        emis += states[s].distribution.parameters
    emis = np.array(emis).reshape(N, -1)

.. note::

    ``tran`` and ``emis`` are estimates of ``tran0`` and ``emis0`` up to a
    switch of rows and columns.

To detection changepoint in this HMM, we need to subclass :class:`AutogradHmm` and
override the method ``loglike_emission``, that is, the
log-likelihood for a normal distribution in our example.

.. code-block:: python

    import math
    import torch
    from autodetect import AutogradHmm
    class MyHMM(AutogradHmm):
        def __init__(self, n_states):
            super(MyHMM, self).__init__(n_states)

        def loglike_emission(self, obs, states):
            emis = self.get_emission()[states, :]
            return -((obs - emis[0]) / emis[1])**2 / 2 -\
                torch.log(torch.tensor(np.sqrt(2 * math.pi)).float() * emis[1])

Finally, we provide the HMM with the MLE of the transition and emission parameters and compute the autograd-test statistic:

.. code-block:: python

    my_hmm = MyHMM(N)
    my_hmm.setup(tran, emis, False)
    obs = torch.from_numpy(y).float()
    stat, tau, index = my_hmm.compute_stats(obs)

If change in emission parameters is of no interest, we can limit the
detection to transition parameters.

.. code-block:: python

    stat, tau, index = my_hmm.compute_stats(obs, idx=range(N * (N - 1)))

.. note::
    For this model both transition and emission parameters are unknown, so
    the indices for transition parameters are :math:`0, \dots, N(N-1)-1`.


API reference
-------------
.. autoclass:: autodetect.AutogradHmm
    :members:
