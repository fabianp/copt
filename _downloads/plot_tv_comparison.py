"""
Total variation regularization
==============================

Comparison of solvers with total variation regularization.
"""
import numpy as np
from scipy import linalg
import pylab as plt
colors = ['#7fc97f', '#beaed4', '#fdc086']


from copt.prox_tv import prox_tv2d, prox_tv1d_rows, prox_tv1d_cols
from copt import davis_yin, proximal_gradient
from copt.utils import Trace, set_mpl_style
from copt.datasets import load_img1


###############################################################
# Load an ground truth image and generate the dataset (A, b) as
#
#             b = A ground_truth + noise   ,
#
# where A is a random matrix
img = load_img1()
n_rows, n_cols = img.shape
n_features = n_rows * n_cols
np.random.seed(0)
n_samples = n_features
#
# plt.imshow(img, interpolation='nearest', cmap=plt.cm.Blues)
# plt.xticks(())
# plt.yticks(())
# plt.show()

# set L2 regularization (arbitrarily) to 1/n_samples
l2_reg = 1.0 / n_samples


A = np.random.uniform(-1, 1, size=(n_samples, n_features))
for i in range(A.shape[0]):
    A[i] /= linalg.norm(A[i])
b = A.dot(img.ravel()) + 1.0 * np.random.randn(n_samples)


def TV(w):
    img = w.reshape((n_rows, n_cols))
    tmp1 = np.abs(np.diff(img, axis=0))
    tmp2 = np.abs(np.diff(img, axis=1))
    return tmp1.sum() + tmp2.sum()


def obj_fun(x):
    return 0.5 * np.linalg.norm(b - A.dot(x)) ** 2 / A.shape[0] + 0.5 * l2_reg * x.dot(x)


def grad(x):
    return - A.T.dot(b - A.dot(x)) / A.shape[0] + l2_reg * x


f, ax = plt.subplots(2, 3, sharey=False)
all_alphas = [1e-6, 1e-3, 1e-1]
for i, alpha in enumerate(all_alphas):

    max_iter = 5000
    trace_three = Trace(lambda x: obj_fun(x) + alpha * TV(x))
    out_tos = davis_yin(
        obj_fun, grad, prox_tv1d_rows, prox_tv1d_cols, np.zeros(n_features),
        alpha=alpha, beta=alpha, g_prox_args=(n_rows, n_cols), h_prox_args=(n_rows, n_cols),
        callback=trace_three, max_iter=max_iter)

    trace_gd = Trace(lambda x: obj_fun(x) + alpha * TV(x))
    out_gd = proximal_gradient(
        obj_fun, grad, prox_tv2d, np.zeros(n_features),
        alpha=alpha, g_prox_args=(n_rows, n_cols, 1000, 1e-1),
        max_iter=max_iter, callback=trace_gd)

    ax[0, i].set_title(r'$\lambda=%s$' % alpha)
    ax[0, i].imshow(out_tos.x.reshape((n_rows, n_cols)),
                    interpolation='nearest', cmap=plt.cm.Blues)
    ax[0, i].set_xticks(())
    ax[0, i].set_yticks(())

    fmin = min(np.min(trace_three.vals), np.min(trace_gd.vals))
    scale = (np.array(trace_three.vals) - fmin)[0]
    ax[1, i].plot(
        np.array(trace_three.times), (np.array(trace_three.vals) - fmin) / scale,
        label='Three operator splitting', lw=4, marker='o', markevery=200,
        markersize=10, color=colors[0])
    ax[1, i].plot(
        np.array(trace_gd.times), (np.array(trace_gd.vals) - fmin) / scale,
        label='ProxGD', lw=4, marker='^', markersize=10, markevery=200,
         color=colors[1])
    # plt.legend(loc='best', frameon=False)
    ax[1, i].set_xlabel('Time (in seconds)')
    ax[1, i].set_yscale('log')
    ax[1, i].grid(True)
plt.legend(ncol=1, frameon=False)
ax[1, 0].set_ylabel('Objective minus optimum')
plt.show()
