API Summary
===========

Observable variable model
-------------------------

For stand-alone models without :class:`torch.nn.Module`:

.. autosummary::

    autodetect.information
    autodetect.autograd_test

For models inherited from :class:`torch.nn.Module`:

.. autosummary::

    autodetect.AutogradTest

Latent variable model
---------------------

.. autosummary::

    autodetect.AutogradHmm

Time series model
-----------------

.. autosummary::

    autodetect.AutogradHmm

Text topic model
----------------

.. autosummary::

    autodetect.AutogradTopic

Data
----

For simulation studies:

.. autosummary::

    autodetect.data.Generator

Utils
-----

For sampling Bessel processes:

.. autosummary::

    autodetect.utils.sample_square_Bessel
    autodetect.utils.sample_max_square_Bessel
    autodetect.utils.quantile_max_square_Bessel