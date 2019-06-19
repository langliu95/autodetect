Time Series Models
==================

A `time series <https://en.wikipedia.org/wiki/Time_series>`_ refers to a series of data points indexed in time order.
A time series model represents the ideal data-generating process of a time series, assuming
the current data point depends on previous ones.
Consequently, computing the log-likelihood function and its derivatives can be time-consuming if the dependency structure is complicated.
Hence, to compute the test statistic more efficiently, algorithms utilize model-specific structure are required.

For `autoregressive--moving-average (ARMA) <https://en.wikipedia.org/wiki/Autoregressive%E2%80%93moving-average_model>`_ models, you can use the method ``autograd_arma`` to detect hidden changes.

Autoregressive--moving-average models
-------------------------------------

Here is an example on how to apply this class to detect changepoint in an ARMA model.
To begin with, let's define functions for generating ARMA parameters with a change
in AR parameters such that the ARMA model is stationary before and after the change.

.. code-block:: python

    import numpy as np
    import numpy.random as npr

    def pars_for_arma(p, q, seed):
        """Generates parameters for ARMA model."""
        npr.seed(seed)
        # generates roots with absolute value larger than 1
        ar_root = npr.exponential(0.5, p) + 1.0
        ar_root *= npr.choice([-1, 1], p)
        ma_root = npr.exponential(0.5, q) + 1.0
        ma_root *= npr.choice([-1, 1], q)
        # extract coefficients
        phi = np.polynomial.polynomial.polyfromroots(ar_root)
        phi /= -phi[0]
        the = np.polynomial.polynomial.polyfromroots(ma_root)
        the /= the[0]
        return phi, the, ar_root, ma_root

    def change_for_ar(phi, ar_root, r):
        """Generates change for AR model."""
        ar_new_root = (1 + r) * ar_root  # add a portion to ar_root
        phi_new = np.polynomial.polynomial.polyfromroots(ar_new_root)
        phi_new /= -phi_new[0]
        delta = phi_new - phi  # post-change - pre-change
        return delta

Now we can generate a synthetic sample from an ARMA model with a changepoint.

.. code-block:: python

    from autodetect.data import Generator
    n, p, q, tau0 = 1000, 3, 2, 500
    seed = 11919
    phi, theta, ar_root, _ = pars_for_arma(p, q, seed)  # parameters before change
    delta = change_for_ar(phi, ar_root, 0.25)  # change in AR parameters
    delta = np.r_[delta, np.zeros(q)]
    gen = Generator(n, 1, tau0)
    y = gen.arma(phi[1:], delta, theta[1:])

Next, we proceed by computing the MLE of parameters under null hypothesis using the package
:class:`statsmodels`.

.. code-block:: python

    import torch
    from statsmodels.tsa.arima_model import ARMA
    cmle = ARMA(y, order=(p, q)).fit(method='css', trend='nc')  # conditional MLE
    mle = torch.tensor(cmle.params).float()  # convert the MLE to a torch.Tensor
    mle.requires_grad = True

Finally, to detection changepoint in this ARMA model, we need to convert all results to `torch.Tensor`
and call the method `autograd_arma`.

.. code-block:: python

    from autodetect import autograd_arma

    obs = torch.from_numpy(np.column_stack([y, np.zeros(n)])).float()
    sig2 = cmle.sigma2
    stat, tau, index = autograd_arma(mle, sig2, obs, p, q)

If change in MA parameters is of no interest, we can limit the
detection to AR parameters.

.. code-block:: python

    stat, tau, index = autograd_arma(mle, sig2, obs, p, q, idx=range(p))

API reference
-------------
.. autofunction:: autodetect.autograd_arma
