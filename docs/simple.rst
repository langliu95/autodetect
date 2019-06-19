Simple Models
=============

In this section we give two examples to show how to apply the autograd-test to detect changepoint in models where all variables are directly observable.
The log-likelihood function can be computed straightforwardly for such models,
which allows an efficient calculation of first derivatives (score function) and
second derivatives (observed information matrix) of the log-likelihood.
These derivatives are computed using the module :class:`torch.autograd`,
so a log-likelihood function (or at least up to an additive constant) implemented in :class:`PyTorch` needs to be provided.
Then you may call the function ``information`` to compute the score function and observed information matrix
at given values of model parameters.

To detect changes in model parameters, the maximum likelihood estimator (MLE) of parameters
under null hypothesis (no change exists) is required.
Once you have the MLE, you can call the function ``autograd_test`` to obtain the test statistic at given significance level,
as well as the location and components of parameters associated with it, that is, the most possible location and components of parameters where the change occurs.
If this statistic is larger than 1, the null hypothesis is rejected,
suggesting it is very likely that there is an abnormal change in your model parameters.


Change in mean
--------------

Here is an example on how to apply this package to detect a changepoint in the mean of a Gaussian model.
To begin with, let's generate some independent multivariate Gaussian random vectors with a sparse change in mean:

.. code-block:: python

    import torch
    from torch.distributions.normal import Normal
    n, d = 1000, 100
    tau0 = int(n / 2)  # location of the change
    theta = torch.zeros(d)  # mean before change
    delta = torch.zeros(d)
    delta[0] = 0.5  # only the first component changes
    gen = Normal(theta, torch.ones(d))
    obs = gen.sample(torch.Size([n]))
    obs[tau0:] += delta

For simplicity, we assume that the covariance matrix is known to be identity matrix.
Then the log-likelihood function (up to a constant) of this model can be defined as

.. code-block:: python

    import numpy as np
    def loglike(theta, obs):
        return -torch.sum((obs - theta[np.newaxis, :])**2) / 2

And the MLE of mean under null hypothesis (no change exists) is

.. code-block:: python

    theta_hat = torch.mean(obs, 0)

Now we can derive the test statistic at significance level 0.05.

.. code-block:: python

    from autodetect import autograd_test
    stat, tau, index = autograd_test(theta_hat, obs, loglike)

If ``stat`` is larger than 1, we reject the null hypothesis.
Moreover, ``tau`` and ``index`` are the location and components of
parameters associated with this test statistic, respectively.


Change in coefficients
----------------------

For models inherited from :class:`torch.nn.Module`, you can utilize the
specifically designed class :class:`AutogradTest`.
For illustration purpose, we consider linear regression.
Firstly, let's generate some linearly correlated observations with change in coefficients:

.. code-block:: python

    import torch
    n, d = 1000, 4
    tau0 = int(n / 2)
    inputs = torch.randn((n, d))
    theta = torch.ones(d)
    delta = 0.5 * torch.ones(d)
    targets = torch.matmul(inputs, theta) + 1 + torch.normal(mean=0.0, std=0.1*torch.ones(n))
    # change in coefficients, not in intercept
    targets[tau0:] += torch.matmul(inputs[tau0:], delta)
    targets = targets.view(-1)

Next we define a linear model as a neural network and its
log-likelihood function (up to a constant).

.. code-block:: python

    import torch.nn as nn
    class Linear(nn.Module):
        """Linear regression as neural network"""
        def __init__(self, in_dim, out_dim):
            super(Linear, self).__init__()
            self.fc = nn.Linear(in_dim, out_dim)

        def forward(self, inputs):
            inputs = self.fc(inputs).view(-1)
            return inputs

    def loglike(outs, targets):
        loss_fn = nn.MSELoss(size_average=False)
        return -loss_fn(outs, targets) / 2

We then train the model with the loss function being negative log-likelihood.

.. code-block:: python

    linear = Linear(d, 1)
    optim = torch.optim.Adam(linear.parameters())
    for _ in range(10000):
        optim.zero_grad()
        outs = linear(inputs)
        loss = -loglike(outs, targets)
        loss.backward()
        optim.step()

Finally, we derive the autograd-test statistic at significance level 0.05.

.. code-block:: python

    from autodetect import AutogradTest
    lin_autograd = AutogradTest(linear, loglike)
    stat, tau, index = lin_autograd.compute_stats(inputs, targets)

Further, if change in intercept is of no interest, you can limit the
detection to coefficients:

.. code-block:: python

    stat, tau, index = lin_autograd.compute_stats(inputs, targets, idx=range(d))

.. note::
    Since coefficients come first in ``linear.parameters()``, the indices for coefficients are :math:`0, \ldots, d-1`.

API reference
-------------

For stand-alone models:

.. autofunction:: autodetect.information

.. autofunction:: autodetect.autograd_test

For models inherited from :class:`torch.nn.Module`:

.. autoclass:: autodetect.AutogradTest
    :members:
