.. score documentation master file, created by
   sphinx-quickstart on Fri Dec 28 22:29:17 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Autodetect
==========

:class:`Autodetect` is a package for a score-based change detection method, *autograd-test*, that can automatically report hidden changes in machine learning systems as they learn from a continuous, possibly evolving, stream of data.
This is code accompanying the paper "`Score-based Change Detection for Gradient-based Learning Machines <https://stat.uw.edu/sites/default/files/2019-07/tr652.pdf>`_".
The package is licensed under the GPLv3 license.

Overview
--------

Given observations :math:`X_1, \dots, X_n` and an underlying probability distribution :math:`p_\theta`
in which :math:`\theta \in \mathbb{R}^d`, this approach detects the existence of a change in
:math:`\theta` through hypothesis testing:

.. math::

    H_0:\ &X_1, \dots, X_n \sim p_\theta \\
    H_1:\ &\exists \tau \in \{1, \ldots, n-1\} \text{ and } \Delta \neq 0 \\
    &s.t.\ X_1, \dots, X_\tau \sim p_\theta \text{ and } X_{\tau+1}, \dots, X_n \sim p_{\theta + \Delta}.

The autograd-test statistic consists of two building blocks---the *linear statistic* and the *scan statistic*.
The scan statistic is especially effective in detecting a *sparse change*, that is, a change that occurs at a small subset of model parameters;
while the linear statistic is more powerful when the change is less sparse.
Both of them are based on the `score statistic <https://en.wikipedia.org/wiki/Score_test>`_, which involves first derivatives (score function) and second derivatives
(observed information matrix) of the log-likelihood function of :math:`\Delta` under null
hypothesis with the maximum likelihood estimator (MLE) of :math:`\theta` plugged in.

This approach also supports detecting changes in a subset of parameters.
For example, one can limit the detection to AR parameters if changes in MA parameters are of no interest for a `autoregressive--moving-average (ARMA) <https://en.wikipedia.org/wiki/Autoregressive%E2%80%93moving-average_model>`_ model.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Getting Started <start.rst>
   API Summary <api.rst>
   Simple Models <simple.rst>
   Latent Variable Models <latent.rst>
   Time Series Models <time.rst>
   Text Topic Models <topic.rst>
   Online Change Detection <autocusum.rst>

Authors
=======

* Lang Liu
* `Joseph Salmon <http://josephsalmon.eu/>`_
* `Zaid Harchaoui <http://faculty.washington.edu/zaid/>`_

Cite
====

If you use this code, please cite::

    @techreport{lsh2019,
    title = {Score-based Change Detection for Gradient-based Learning Machines},
    author = {Liu, Lang and
              Salmon, Joseph and
              Harchaoui, Zaid},
    year = {2019},
    institution = {Department of Statistics, University of Washington},
    month = {June}
    }

Acknowledgments
---------------
This work was supported by NSF CCF-1740551, NSF DMS-1810975, the program “Learning in
Machines and Brains” of CIFAR, and faculty research awards.

|adsi|_
|nbsp| |nbsp| |nbsp| |nbsp|
|nsf|_
|esci|_
|nbsp| |nbsp|
|nbsp| |nbsp| |nbsp| |nbsp|
|cifar|_

.. |adsi| image:: fig/ADSI.png
   :width: 25%

.. _adsi: http://ads-institute.uw.edu/

.. |nsf| image:: fig/NSF.png
   :width: 11%

.. _nsf: https://nsf.gov/

.. |esci| image:: fig/escience.png
    :width: 40%

.. _esci: https://escience.washington.edu/

.. |cifar| image:: fig/CIFAR.png
    :width: 15%

.. _cifar: https://www.cifar.ca/

.. |nbsp| unicode:: 0xA0
   :trim:


.. Indices and Tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
