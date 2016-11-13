"""
L1-regularized logistic regression
==================================

Implementation of L1-regularized logistic regression
using copt.
"""
import numpy as np
from sklearn.linear_model import logistic
from copt import proximal_gradient

n_samples, n_features = 100, 10
X = np.random.randn(n_samples, n_features)
y = np.random.randn(n_samples)
alpha = 1.


def logloss(x):
    return logistic._logistic_loss(x, X, y, 1.)


def fprime_logloss(x):
    return logistic._logistic_loss_and_grad(x, X, y, 1.)[1]


def g_prox(x, step_size):
    """
    prox of alpha * l1
    """
    return np.fmax(x - step_size * alpha, 0) - \
        np.fmax(- x - step_size * alpha, 0)


out = proximal_gradient(logloss, fprime_logloss, g_prox, np.zeros(n_features))
print('Solution', out)
