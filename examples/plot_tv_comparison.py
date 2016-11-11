"""
Total variation regularization
==============================

Comparison of solvers with total variation regularization.
"""
import numpy as np
from scipy import linalg
from copt.prox_tv import prox_tv2d, prox_tv1d_rows, prox_tv1d_cols
from copt import three_split, proximal_gradient
from copt.utils import Trace
from copt.datasets import load_img1
import pylab as plt

plt.style.use('fivethirtyeight')


img = load_img1()

##############################################################
# generate measurements as
#             b = A ground_truth + noise   ,
# where X is a random matrix
n_rows, n_cols = img.shape
n_features = n_rows * n_cols
np.random.seed(0)
n_samples = n_features


A = np.random.uniform(-1, 1, size=(n_samples, n_features))
for i in range(A.shape[0]):
    A[i] /= linalg.norm(A[i])
b = A.dot(img.ravel()) + 1.0 * np.random.randn(n_samples)


def TV(w):
    img = w.reshape((n_rows, n_cols))
    tmp1 = np.abs(np.diff(img, axis=0))
    tmp2 = np.abs(np.diff(img, axis=1))
    return tmp1.sum() + tmp2.sum()

l2_reg = 1.0 / n_samples

def obj_fun(x):
    return 0.5 * np.linalg.norm(b - A.dot(x)) ** 2 / A.shape[0] + 0.5 * l2_reg * x.dot(x)

def grad(x):
    return - A.T.dot(b - A.dot(x)) / A.shape[0] + l2_reg * x

############################
# Find best step size

from lightning.impl.sag import get_auto_step_size, get_dataset
ds = get_dataset(A, order="c")
step_size = get_auto_step_size(ds, l2_reg, 'squared')

all_alphas = [1e-6, 1e-3, 1e-1]
for alpha in all_alphas:

    max_iter = 50000
    backtracking = False
    trace_three = Trace(lambda x: obj_fun(x) + alpha * TV(x))
    out_tos = three_split(
        obj_fun, grad, prox_tv1d_rows, prox_tv1d_cols, np.zeros(n_features),
        alpha=alpha, beta=alpha, step_size=step_size,
        g_prox_args=(n_rows, n_cols), h_prox_args=(n_rows, n_cols),
        callback=trace_three, max_iter=max_iter, tol=1e-10, backtracking=backtracking)

    trace_gd = Trace(lambda x: obj_fun(x) + alpha * TV(x))
    out_gd = proximal_gradient(
        obj_fun, grad, prox_tv2d, np.zeros(n_features),
        alpha=alpha, g_prox_args=(n_rows, n_cols, 1000, 1e-1),
        step_size=step_size, max_iter=max_iter, tol=1e-10,
        backtracking=backtracking, callback=trace_gd)

    plt.matshow(out_gd.x.reshape((n_rows, n_cols)))
    plt.show()
    # plotting code
    fmin = min(np.min(trace_three.vals), np.min(trace_gd.vals))
    scale = (np.array(trace_three.vals) - fmin)[0]
    plt.figure()
    plt.title(r'$\lambda=%s$' % alpha)
    plt.plot(np.array(trace_three.times),
             (np.array(trace_three.vals) - fmin) / scale,
             label='Three operator splitting', lw=4, marker='o',
             markevery=500)
    plt.plot(np.array(trace_gd.times),
             (np.array(trace_gd.vals) - fmin) / scale, label='ProxGD',
             lw=4, marker='h', markevery=500)
    plt.legend()
    plt.xlabel('Time (in seconds)')
    plt.yscale('log')
    plt.show()