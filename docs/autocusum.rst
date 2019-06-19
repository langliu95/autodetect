Online Change Detection
=======================

In some situations data may arrive continually (or the full dataset is too large to be stored in memory),
so an algorithm which continuously inspects the machine learning system as new data arrive is desirable.
This package contains an online variant of the autograd-test, called *autograd-test-CuSum*.

In consideration of computational time, this approach is only implemented for independent models, that is, it assumes observations in the dataset are independent.
In addition, online approximation is employed in order to compute the statistic in a linear (in sample size) time.
Consequently, this algorithm may perform poorly when the number of parameters is large.

To control the false discovery rate, a sample size indicating the total number of observations used for change detection is required.
After that number of observations, you may reinitialize the algorithm and run it on new data.

Autograd-test-CuSum
-------------------

For illustration purpose, let's generate some observations from a linear regression model.

.. code-block:: python

    import torch
    n = 10000
    train_size = 3000
    dim = 10
    tau = 6000 + train_size
    total = n + train_size
    # generates observations
    inputs = torch.randn((total, dim))
    targets = torch.randn(total)
    delta = torch.zeros(dim)
    delta[0] = 0.5
    targets[tau:] += torch.matmul(inputs[tau:], delta)

Then we train a linear model using the first ``train_size`` observations.

.. code-block:: python

    import torch.nn as nn
    from autodetect import AutogradCuSum
    from autodetect.data import Linear
    def loglike(out, tar):
        loss_fn = nn.MSELoss(size_average=False)
        return -loss_fn(out, tar) / 2

    linear = Linear(dim, 1)
    optim = torch.optim.Adam(linear.parameters())
    for _ in range(10000):
        optim.zero_grad()
        outs = linear(inputs[:train_size])
        loss = -loglike(outs, targets[:train_size])
        loss.backward()
        optim.step()
    autocusum = AutogradCuSum(linear, loglike)

Different from autograd-test, we need to compute the threshold before calling the algorithm,
where the threshold for the linear test at significance level :math:`\alpha` is the upper :math:`(1-\alpha)`-quantile of the maximum of the square `Bessel process <https://en.wikipedia.org/wiki/Bessel_process>`_.
Here we utilize the algorithm described in the book `"Monte Carlo Methods in Financial Engineering" <https://www.springer.com/us/book/9780387004518>`_ to directly sample from the maximum of the sqaure Bessel process, which is implemented in the function ``quantile_max_square_Bessel``.

.. code-block:: python

    from autodetect.utils import quantile_max_square_Bessel
    thresh = quantile_max_square_Bessel(0.95, dim+1, 2000, 5000)  # dim+1 for intercept

Next, we run it on the first half sample.

.. code-block:: python

    size1 = 5000 + train_size  # include train_size
    autocusum.initial_model(inputs[:train_size], targets[:train_size], size1)
    for i in range(train_size, size1):  # feed one obs at a time
        stat = autocusum.compute_stats(inputs[i:(i+1)], targets[i:(i+1)], thresh[-1])
        if stat > 1.0:
            rej_length = i - train_size
            print(f"rejection at the {rej_length}-th new observation.")
            break

Finally, we reinitialize the algorithm and run it on the second half sample.

.. code-block:: python

    size2 = total - size1 + train_size
    autocusum.initial_model(inputs[:train_size], targets[:train_size], size2)
    for i in range(size1, total):  # feed one obs at a time
        stat = autocusum.compute_stats(inputs[i:(i+1)], targets[i:(i+1)], thresh[-1])
        if stat > 1.0:
            rej_length = i - size1
            print(f"rejection at the {rej_length}-th new observation.")
            break


API reference
-------------

For threshold:

.. autofunction:: autodetect.utils.sample_square_Bessel

.. autofunction:: autodetect.utils.sample_max_square_Bessel

.. autofunction:: autodetect.utils.quantile_max_square_Bessel

For autograd-test-CuSum:

.. autoclass:: autodetect.AutogradCuSum
    :members:
