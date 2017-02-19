from concurrent import futures
from typing import Callable
from datetime import datetime
import numpy as np
from scipy import sparse, optimize
from numba import njit
from copt.utils import norm_rows

import concurrent.futures

@njit
def f_squared(p, y):
    # squared loss
    return 0.5 * ((y - p) ** 2)


@njit
def deriv_squared(p, y):
    # derivative of squared loss
    return - (y - p)


@njit
def f_logistic(p, y):
    # logistic loss
    # same as in lightning
    p *= y
    if p > 0:
        return np.log(1 + np.exp(-p))
    else:
        return -p + np.log(1 + np.exp(p))


@njit
def deriv_logistic(p, y):
    # derivative of logistic loss
    # same as in lightning (with minus sign)
    p *= y
    if p > 0:
        phi = 1. / (1 + np.exp(-p))
    else:
        exp_t = np.exp(p)
        phi = exp_t / (1. + exp_t)
    return (phi - 1) * y


@njit
def prox_L1(step_size: float, x: np.ndarray, low: int, high: int):
    """
    L1 proximal operator. Inplace.
    """
    for j in range(low, high):
        x[j] = np.fmax(x[j] - step_size, 0) - np.fmax(- x[j] - step_size, 0)


@njit
def f_L1(x):
    return np.sum(np.abs(x))


def compute_step_size(loss: str, A, alpha: float, step_size_factor=4) -> float:
    """
    Helper function to compute the step size for common loss
    functions.

    Parameters
    ----------
    loss
    A
    step_size_factor

    Returns
    -------

    """
    if loss == 'logistic':
        L = 0.25 * norm_rows(A) + alpha
        return (1.0 / L) / step_size_factor
    elif loss == 'squared':
        L = norm_rows(A) + alpha
        return (1.0 / L) / step_size_factor
    else:
        raise NotImplementedError('loss %s is not implemented' % loss)


def fmin_SAGA(
        fun: Callable, fun_deriv: Callable, A, b, x0: np.ndarray,
        alpha: float=0., beta: float=0., g_prox: Callable=None, step_size: float=-1,
        g_func: Callable=None,
        g_blocks: np.ndarray=None, n_jobs: int=1, max_iter=100, tol=1e-6,
        verbose=False, callback=None, trace=False) -> optimize.OptimizeResult:
    """Stochastic average gradient augmented (SAGA) algorithm.

    The SAGA algorithm can solve optimization problems of the form

        argmin_x 1/n \sum_{i=1}^n f(a_i^T x, b_i) + alpha * L2 + beta * g(x)


    Parameters
    ----------
    fun
        loss function

    fun_deriv
        derivative function

    alpha
        Amount of squared L2 regularization

    x0
        Starting point

    g_blocks
        If g is a block-separable function, this allows to specify which are the
        blocks in this penalty. It is an array of integers with the same size as
        x0 where each coordinate represents the group to which that coordinate
        belongs to.

    Returns
    -------
    opt
        The optimization result represented as a
        ``scipy.optimize.OptimizeResult`` object. Important attributes are:
        ``x`` the solution array, ``success`` a Boolean flag indicating if
        the optimizer exited successfully and ``message`` which describes
        the cause of the termination. See `scipy.optimize.OptimizeResult`
        for a description of other attributes.

    References
    ----------
    Defazio, Aaron, Francis Bach, and Simon Lacoste-Julien. "SAGA: A fast
    incremental gradient method with support for non-strongly convex composite
    objectives." Advances in Neural Information Processing Systems. 2014.
    """

    x = np.ascontiguousarray(x0).copy()
    assert x.size == A.shape[1]
    assert A.shape[0] == b.size

    if step_size < 0:
        raise ValueError

    # TODO: encapsulate this in _get_factory
    if hasattr(g_prox, '__call__'):
        if not hasattr(g_prox, 'inspect_llvm'):
            g_prox = njit(g_prox)
    elif g_prox is None:
        @njit
        def g_prox(step_size, x, *args): return x
    else:
        raise NotImplementedError

    if g_func is None:
        @njit
        def g_func(x, *args):
            return 0

    n_samples, n_features = A.shape
    success = False

    A = sparse.csr_matrix(A)
    if g_blocks is None:
        g_blocks = np.zeros(n_features, dtype=np.int64)
    epoch_iteration, init_gradient, trace_loss = _epoch_factory_sparse_SAGA(
            fun, g_func, fun_deriv, g_prox, g_blocks, A, b, alpha, beta)

    # .. memory terms ..
    memory_gradient = np.zeros(n_samples)
    gradient_average = np.zeros(n_features)
    init_gradient(memory_gradient, gradient_average, n_samples)

    # warm up for the JIT
    epoch_iteration(
        x.copy(), memory_gradient, gradient_average, np.array([0]), step_size)

    trace_func = []
    start_time = datetime.now()
    trace_time = [0.]
    trace_x = [x.copy()]
    trace_certificate = [np.inf]

    # .. iterate on epochs ..
    for it in range(max_iter):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for _ in range(n_jobs):
                perm = np.random.randint(low=0, high=n_samples, size=n_samples)
                futures.append(executor.submit(
                    epoch_iteration, x, memory_gradient, gradient_average,
                    perm, step_size))
            concurrent.futures.wait(futures)

        grad = gradient_average + alpha * x
        z = x - step_size * grad
        g_prox(beta * step_size, z, 0, n_features)
        certificate = np.linalg.norm(x - z)
        if callback is not None:
            callback(x)
        if trace:
            trace_x.append(x.copy())
            trace_certificate.append(certificate)
            trace_time.append((datetime.now() - start_time).total_seconds())

        if verbose:
            print('Iteration: %s, certificate: %s' % (it, certificate))
        if certificate < tol:
            success = True
            break
    if trace:
        print('Computing trace')
        # .. compute function values ..
        trace_func = []
        for i in range(len(trace_x)):
            trace_func.append(trace_loss(trace_x[i]))

    return optimize.OptimizeResult(
        x=x, success=success, nit=it, trace_func=trace_func, trace_time=trace_time,
        certificate=certificate, trace_certificate=trace_certificate)


def fmin_PSSAGA(
        fun, fun_deriv, A, b, x0, g_prox=None, h_prox=None,
        alpha: float=0.0,
        beta: float=0.0,
        gamma: float=0.0,
        step_size=-1,
        g_func=None,
        h_func=None,
        h_blocks=None,
        max_iter=100, tol=1e-6, verbose=False, callback=None, trace=False):

    n_samples, n_features = A.shape
    success = False

    if hasattr(g_prox, '__call__'):
        pass
        if not hasattr(g_prox, 'inspect_llvm'):
            g_prox = njit(g_prox)
    elif g_prox is None:
        @njit
        def g_prox(step_size, x, y, low, high, weights):
            for i in range(low, high):
                y[i] = x[i]
    else:
        raise NotImplementedError

    if hasattr(h_prox, '__call__'):
        pass
        if not hasattr(h_prox, 'inspect_llvm'):
            # if it has not yet been jitted
            h_prox = njit(h_prox)
    elif h_prox is None:
        @njit
        def h_prox(step_size, x, *args): pass
        if h_blocks is None:
             h_blocks = np.arange(n_features)
    else:
        raise NotImplementedError

    # .. define g and h if not done already ..
    if g_func is None:
        @njit
        def g_func(*args): return 0
    if h_func is None:
        @njit
        def h_func(*args): return 0

    y = x0.copy()
    x = x0.copy()
    z = x.copy()

    # assert y.shape[1] == A.shape[1]
    assert A.shape[0] == b.size

    if step_size < 0:
        raise ValueError

    A = sparse.csr_matrix(A)
    if h_blocks is None:
        h_blocks = np.zeros(n_features, dtype=np.int64)
    epoch_iteration, init_gradient, trace_loss = _factory_sparse_PSSAGA(
        fun, g_func, h_func, fun_deriv, g_prox, h_prox, h_blocks, A, b,
        alpha, beta, gamma)

    # .. memory terms ..
    memory_gradient = np.zeros(n_samples)
    gradient_average = np.zeros(n_features)
    init_gradient(memory_gradient, gradient_average, n_samples)

    # warm up for the JIT
    epoch_iteration(
        y, x, z, memory_gradient, gradient_average, np.array([0]),
        step_size)

    trace_func = []
    start_time = datetime.now()
    trace_time = [0.]
    trace_x = [x.copy()]
    trace_certificate = [np.inf]

    # .. iterate on epochs ..
    for it in range(max_iter):
        epoch_iteration(
            y, x, z, memory_gradient, gradient_average, np.random.permutation(n_samples),
            step_size)

        certificate = np.linalg.norm(x - z)
        if callback is not None:
            callback(x)
        if trace:
            trace_x.append(x.copy())
            trace_certificate.append(certificate)
            trace_time.append((datetime.now() - start_time).total_seconds())
        if verbose:
            print('Iteration: %s, certificate: %s' % (it, certificate))
        if certificate < tol:
            success = True
            break
    if trace:
        if verbose:
            print('.. computing trace ..')
        # .. compute function values ..
        trace_func = []
        for i in range(len(trace_x)):
            trace_func.append(trace_loss(trace_x[i]))

    return optimize.OptimizeResult(
        x=x, y=y, z=z, success=success, nit=it, trace_x=trace_x,
        certificate=certificate,
        trace_func=trace_func, trace_certificate=np.array(trace_certificate),
        trace_time=trace_time)


@njit(nogil=True, cache=True)
def _support_matrix(
        A_indices, A_indptr, g_blocks, n_blocks):
    """
    """
    if n_blocks == 1:
        # XXX FIXME do something smart
        pass
    BS_indices = np.zeros(A_indices.size, dtype=np.int64)
    BS_indptr = np.zeros(A_indptr.size, dtype=np.int64)
    seen_blocks = np.zeros(n_blocks, dtype=np.int64)
    BS_indptr[0] = 0
    counter_indptr = 0
    for i in range(A_indptr.size - 1):
        low = A_indptr[i]
        high = A_indptr[i + 1]
        for j in range(low, high):
            g_idx = g_blocks[A_indices[j]]
            if seen_blocks[g_idx] == 0:
                # if first time we encouter this block,
                # add to the index and mark as seen
                BS_indices[counter_indptr] = g_idx
                seen_blocks[g_idx] = 1
                counter_indptr += 1
        BS_indptr[i+1] = counter_indptr
        # cleanup
        for j in range(BS_indptr[i], counter_indptr):
            seen_blocks[BS_indices[j]] = 0
    BS_data = np.ones(counter_indptr)
    return BS_data, BS_indices[:counter_indptr], BS_indptr


def _epoch_factory_sparse_SAGA(
        f_func, g_func, f_prime, g_prox, g_blocks, A, b, alpha, beta):

    A_data = A.data
    A_indices = A.indices
    A_indptr = A.indptr
    n_samples, n_features = A.shape

    # g_blocks is a map from n_features -> n_features
    unique_blocks = np.unique(g_blocks)
    n_blocks = np.unique(g_blocks).size
    assert np.all(unique_blocks == np.arange(n_blocks))

    # .. compute the block support ..
    BS_data, BS_indices, BS_indptr = _support_matrix(
        A_indices, A_indptr, g_blocks, n_blocks)
    BS = sparse.csr_matrix((BS_data, BS_indices, BS_indptr), (n_samples, n_blocks))

    # .. estimate a mapping from blocks to features ..
    reverse_blocks = sparse.dok_matrix((n_blocks, n_features), dtype=np.bool)
    for j_feat in range(n_features):
        i_block = g_blocks[j_feat]
        reverse_blocks[i_block, j_feat] = True
    reverse_blocks = reverse_blocks.tocsr()
    RB_indptr = reverse_blocks.indptr

    d = np.array(BS.sum(0), dtype=np.float).ravel()
    idx = (d != 0)
    d[idx] = n_samples / d[idx]

    @njit
    def init_gradient(memory_gradient, gradient_average, n_samples):
        for i in range(n_samples):

            grad_i = f_prime(0., b[i])
            memory_gradient[i] = grad_i

            for j in range(A_indptr[i], A_indptr[i + 1]):
                j_idx = A_indices[j]
                gradient_average[j_idx] += grad_i * A_data[j] / n_samples


    @njit(nogil=True)
    def epoch_iteration_template(
            x, memory_gradient, gradient_average, sample_indices, step_size):

        # .. SAGA estimate of the gradient ..
        # incr = np.zeros(n_features, dtype=x.dtype)

        # .. inner iteration ..
        for i in sample_indices:

            p = 0.
            for j in range(A_indptr[i], A_indptr[i+1]):
                j_idx = A_indices[j]
                p += x[j_idx] * A_data[j]

            grad_i = f_prime(p, b[i])
            mem_i = memory_gradient[i]

            # .. update coefficients ..
            for j in range(A_indptr[i], A_indptr[i+1]):
                j_idx = A_indices[j]
                x_j = x[j_idx]
                incr = (grad_i - mem_i) * A_data[j] + (
                    gradient_average[j_idx] + alpha * x_j)

                x_new = x_j - step_size * incr
                x[j_idx] = x_new

            g_prox(step_size * beta, x, 0, n_features)

                #
                # for b_j in range(RB_indptr[g], RB_indptr[g+1]):
                #     # update vector of coefficients
                #     tmp = incr[b_j] - x[b_j]
                #     x[b_j] += tmp
                #     # x[b_j] = incr[b_j]
                #     incr[b_j] = 0

            # .. update memory terms ..
            for j in range(A_indptr[i], A_indptr[i+1]):
                j_idx = A_indices[j]
                gradient_average[j_idx] += (grad_i - memory_gradient[i]) * A_data[j] / n_samples
            memory_gradient[i] += (grad_i - mem_i)

    @njit
    def full_loss(x):
        obj = 0.
        for i in range(n_samples):
            idx = A_indices[A_indptr[i]:A_indptr[i + 1]]
            A_i = A_data[A_indptr[i]:A_indptr[i + 1]]
            obj += f_func(np.dot(x[idx], A_i), b[i]) / n_samples
        return obj + 0.5 * alpha * np.dot(x, x) + beta * g_func(x)

    return epoch_iteration_template, init_gradient, full_loss


def _factory_sparse_PSSAGA(
        fun, g_func, h_func, f_prime, g_prox, h_prox, h_blocks, A, b, alpha,
        beta, gamma):

    A_data = A.data
    A_indices = A.indices
    A_indptr = A.indptr
    n_samples, n_features = A.shape

    # g_blocks is a map from n_features -> n_features
    unique_blocks_h = np.unique(h_blocks)
    n_blocks_h = np.unique(h_blocks).size
    assert np.all(unique_blocks_h == np.arange(n_blocks_h))

    BS_h_data, BS_h_indices, BS_h_indptr = _support_matrix(
        A_indices, A_indptr, h_blocks, n_blocks_h)
    BS_h = sparse.csr_matrix((BS_h_data, BS_h_indices, BS_h_indptr))

    # .. estimate a mapping from blocks to features ..
    reverse_blocks_h = sparse.dok_matrix((n_blocks_h, n_features), dtype=np.bool)
    for j_ in range(n_features):
        i_ = h_blocks[j_]
        reverse_blocks_h[i_, j_] = True
    reverse_blocks_h = reverse_blocks_h.tocsr()
    RB_h_indptr = reverse_blocks_h.indptr

    d_h = np.array(BS_h.sum(0), dtype=np.float).ravel()
    idx = (d_h != 0)
    d_h[idx] = n_samples / d_h[idx]
    d_h[~idx] = 1

    d_weights = np.zeros(n_features)
    for h in range(n_blocks_h):
        for b_j in range(RB_h_indptr[h], RB_h_indptr[h + 1]):
            d_weights[b_j] = 1.0 / d_h[h]

    @njit
    def init_gradient(memory_gradient, gradient_average, n_samples):
        for i in range(n_samples):

            grad_i = f_prime(0., b[i])
            memory_gradient[i] = grad_i

            for j in range(A_indptr[i], A_indptr[i + 1]):
                j_idx = A_indices[j]
                gradient_average[j_idx] += grad_i * A_data[j] / n_samples


    @njit
    def epoch_iteration_template(
            y, x, z, memory_gradient, gradient_average, sample_indices, step_size):

        # .. SAGA estimate of the gradient ..
        grad_est = np.zeros(n_features)

        # .. iterate on samples ..
        for i in sample_indices:

            for h_j in range(BS_h_indptr[i], BS_h_indptr[i+1]):
                h = BS_h_indices[h_j]
                g_prox(step_size * beta, y, x, RB_h_indptr[h], RB_h_indptr[h+1], d_weights)

            p = 0.
            for j in range(A_indptr[i], A_indptr[i+1]):
                j_idx = A_indices[j]
                p += x[j_idx] * A_data[j]

            grad_i = f_prime(p, b[i])

            # .. gradient estimate (XXX difference) ..
            for j in range(A_indptr[i], A_indptr[i+1]):
                j_idx = A_indices[j]
                grad_est[j_idx] = (grad_i - memory_gradient[i]) * A_data[j]

            # .. iterate on blocks ..
            for h_j in range(BS_h_indptr[i], BS_h_indptr[i+1]):
                h = BS_h_indices[h_j]

                # .. iterate on features inside block ..
                for b_j in range(RB_h_indptr[h], RB_h_indptr[h+1]):
                    bias_term = d_h[h] * (gradient_average[b_j] + alpha * x[b_j])
                    z[b_j] = 2 * x[b_j] - y[b_j] - step_size * (
                        grad_est[b_j] + bias_term)

                h_prox(d_h[h] * step_size * gamma, z, RB_h_indptr[h], RB_h_indptr[h+1])

                # .. update y ..
                for b_j in range(RB_h_indptr[h], RB_h_indptr[h+1]):
                    y[b_j] -= x[b_j] - z[b_j]

            # .. update memory terms ..
            for j in range(A_indptr[i], A_indptr[i+1]):
                j_idx = A_indices[j]
                gradient_average[j_idx] += (grad_i - memory_gradient[i]) * A_data[j] / n_samples
                grad_est[j_idx] = 0
            memory_gradient[i] = grad_i

    @njit
    def full_loss(x):
        obj = 0.
        for i in range(n_samples):
            idx = A_indices[A_indptr[i]:A_indptr[i + 1]]
            A_i = A_data[A_indptr[i]:A_indptr[i + 1]]
            obj += fun(np.dot(x[idx], A_i), b[i]) / n_samples
        obj += 0.5 * alpha * np.dot(x, x) + beta * g_func(x) + gamma * h_func(x)
        return obj

    return epoch_iteration_template, init_gradient, full_loss
