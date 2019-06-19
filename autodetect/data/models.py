"""Models in PyTorch.

Author: Lang Liu
Date: 06/10/2019
"""

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn


def loglike_gaussian(theta, obs, sig=1.0):
    """Log-likelihood for Gaussian model (without constant term)."""
    return -torch.sum((obs - theta[np.newaxis, :])**2) / 2 / sig**2


def loglike_linear(theta, data):
    """Log-likelihood for linear regression."""
    x, y = data[:, :-1], data[:, -1]
    y_pred = x @ theta
    criterion = nn.MSELoss(size_average=False)
    loglike = -criterion(y_pred, y.view(-1)) / 2
    return loglike


class Linear(nn.Module):
    """Linear regression as a neural network."""
    def __init__(self, in_dim, out_dim):
        super(Linear, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, inputs):
        inputs = self.fc(inputs).view(-1)
        return inputs


# for ARMA model
# define the order here
P, Q = 3, 2
LAG = max(P, Q)

def loglike_arma(theta, data):
    """Log-likelihood for ARMA model.

    This log-likelihood is computed by conditioning on :math:`x_0 = \\cdots =
    x_{-p+1} = \\epsilon_0 = \\cdots = \\epsilon_{-q+1} = 0`.

    Residuals in data will be updated (``data[LAG:, 1]``).

    Parameters
    ----------
    theta:  torch.Tensor (P+Q, )
        ARMA coefficients and mean and SD of white noise.
    data: torch.Tensor, shape (size+LAG, 2)
        The first column is observation and the second column is residual.
    """

    x, e = data[:, 0], data[:, 1]
    length = len(x) - LAG
    resi = torch.zeros(length, requires_grad=True)
    for t in range(LAG, len(x)):
        inv_idx = torch.arange(t-1, t-P-1, -1).long()
        inv_ide = torch.arange(t-1, t-Q-1, -1).long()
        resi[t-LAG] = x[t] - theta[-2] - torch.sum(theta[:P] * x[inv_idx]) -\
            torch.sum(theta[P:(P+Q)] * e[inv_ide])
        e[t] = resi[t-LAG].detach()
    loglike = -length * torch.log(theta[-1]) - \
        torch.sum(resi**2) / theta[-1]**2 / 2
    return loglike
