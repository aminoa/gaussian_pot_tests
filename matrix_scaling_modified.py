# stock implementation of Sinkhorn/Greenkhorn based on POT library

import warnings

import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from ot.utils import unif, dist, list_to_array

from ot.backend import get_backend

import random

def sinkhorn_log_modified(a, b, M, reg, numItermax=1000, stopThr=1e-9, verbose=False, log=False, warn=True, **kwargs):
    a, b, M = list_to_array(a, b, M)
    nx = get_backend(M, a, b)

    if len(a) == 0:
        a = nx.full((M.shape[0],), 1.0 / M.shape[0], type_as=M)
    if len(b) == 0:
        b = nx.full((M.shape[1],), 1.0 / M.shape[1], type_as=M)

    # init data
    dim_a = len(a)
    dim_b = b.shape[0]

    if len(b.shape) > 1:
        n_hists = b.shape[1]
    else:
        n_hists = 0

    if n_hists:  # we do not want to use tensors sor we do a loop

        lst_loss = []
        lst_u = []
        lst_v = []

        for k in range(n_hists):
            res = sinkhorn_log_modified(a, b[:, k], M, reg, numItermax=numItermax, stopThr=stopThr, verbose=verbose, log=log, **kwargs)

            if log:
                lst_loss.append(nx.sum(M * res[0]))
                lst_u.append(res[1]['log_u'])
                lst_v.append(res[1]['log_v'])
            else:
                lst_loss.append(nx.sum(M * res))
        res = nx.stack(lst_loss)
        if log:
            log = {'log_u': nx.stack(lst_u, 1),
                   'log_v': nx.stack(lst_v, 1), }
            log['u'] = nx.exp(log['log_u'])
            log['v'] = nx.exp(log['log_v'])
            return res, log
        else:
            return res
    else:
        if log:
            log = {'err': []}

        Mr = - M / reg
        u = nx.zeros(dim_a, type_as=M)
        v = nx.zeros(dim_b, type_as=M)

        def get_logT(u, v):
            if n_hists:
                return Mr[:, :, None] + u + v
            else:
                return Mr + u[:, None] + v[None, :]

        loga = nx.log(a)
        logb = nx.log(b)

        err = 1
        for ii in range(numItermax):

            v = logb - nx.logsumexp(Mr + u[:, None], 0)
            u = loga - nx.logsumexp(Mr + v[None, :], 1)

            if ii % 10 == 0:
                tmp2 = nx.sum(nx.exp(get_logT(u, v)), 0)
                err = nx.norm(tmp2 - b)  # violation of marginal
                if log:
                    log['err'].append(err)

                if verbose:
                    if ii % 200 == 0:
                        print(
                            '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                    print('{:5d}|{:8e}|'.format(ii, err))
                if err < stopThr:
                    break
        else:
            if warn:
                warnings.warn("Sinkhorn did not converge. You might want to "
                              "increase the number of iterations `numItermax` "
                              "or the regularization parameter `reg`.")

        if log:
            log['niter'] = ii
            log['log_u'] = u
            log['log_v'] = v
            log['u'] = nx.exp(u)
            log['v'] = nx.exp(v)

            return nx.exp(get_logT(u, v)), log

        else:
            return nx.exp(get_logT(u, v))

def greenkhorn_inverse_gauss(a, b, M, reg, numItermax=10000, stopThr=1e-9, verbose=False, log=False, warn=True):
    a, b, M = list_to_array(a, b, M)

    nx = get_backend(M, a, b)
    if nx.__name__ in ("jax", "tf"):
        raise TypeError("JAX or TF arrays have been received. Greenkhorn is not "
                        "compatible with  neither JAX nor TF")

    if len(a) == 0:
        a = nx.ones((M.shape[0],), type_as=M) / M.shape[0]
    if len(b) == 0:
        b = nx.ones((M.shape[1],), type_as=M) / M.shape[1]

    dim_a = a.shape[0]
    dim_b = b.shape[0]

    K = nx.exp(-M / reg)

    u = nx.full((dim_a,), 1. / dim_a, type_as=K)
    v = nx.full((dim_b,), 1. / dim_b, type_as=K)
    G = u[:, None] * K * v[None, :]

    viol = nx.sum(G, axis=1) - a # compresses all the columns into (by adding all the values in each row into one column vector 
    viol_2 = nx.sum(G, axis=0) - b #compress all the rows by summing all the valeus across each column into one  row vector
    stopThr_val = 1
    if log:
        log = dict()
        log['u'] = u
        log['v'] = v

    for ii in range(numItermax):
        # i_1 = nx.argmax(nx.abs(viol))
        # i_2 = nx.argmax(nx.abs(viol_2))

        # this isn't an inverse gaussian - it outputs values opposite of a gaussian 
        i_1 = int(min(len(viol) - 1, max(0, 1 - random.gauss(len(viol) / 2, len(viol) / 2))))
        i_2 = int(min(len(viol_2) - 1, max(0, 1 - random.gauss(len(viol_2) / 2, len(viol_2) / 2))))
        # i_2 = int(random.randrange(0, len(viol_2))) 

        m_viol_1 = nx.abs(viol[i_1])
        m_viol_2 = nx.abs(viol_2[i_2])
        stopThr_val = nx.maximum(m_viol_1, m_viol_2)

        if m_viol_1 > m_viol_2:
            old_u = u[i_1]
            new_u = a[i_1] / nx.dot(K[i_1, :], v)
            G[i_1, :] = new_u * K[i_1, :] * v

            viol[i_1] = nx.dot(new_u * K[i_1, :], v) - a[i_1]
            viol_2 += (K[i_1, :].T * (new_u - old_u) * v)
            u[i_1] = new_u
        else:
            old_v = v[i_2]
            new_v = b[i_2] / nx.dot(K[:, i_2].T, u)
            G[:, i_2] = u * K[:, i_2] * new_v
            # aviol = (G@one_m - a)
            # aviol_2 = (G.T@one_n - b)
            viol += (-old_v + new_v) * K[:, i_2] * u
            viol_2[i_2] = new_v * nx.dot(K[:, i_2], u) - b[i_2]
            v[i_2] = new_v

        if stopThr_val <= stopThr:
            break
    else:
        if warn:
            warnings.warn("Sinkhorn did not converge. You might want to "
                          "increase the number of iterations `numItermax` "
                          "or the regularization parameter `reg`.")

    if log:
        log["n_iter"] = ii
        log['u'] = u
        log['v'] = v

    if log:
        return G, log
    else:
        return G

def greenkhorn_random(a, b, M, reg, numItermax=10000, stopThr=1e-9, verbose=False, log=False, warn=True):
    a, b, M = list_to_array(a, b, M)

    nx = get_backend(M, a, b)
    if nx.__name__ in ("jax", "tf"):
        raise TypeError("JAX or TF arrays have been received. Greenkhorn is not "
                        "compatible with  neither JAX nor TF")

    if len(a) == 0:
        a = nx.ones((M.shape[0],), type_as=M) / M.shape[0]
    if len(b) == 0:
        b = nx.ones((M.shape[1],), type_as=M) / M.shape[1]

    dim_a = a.shape[0]
    dim_b = b.shape[0]

    K = nx.exp(-M / reg)

    u = nx.full((dim_a,), 1. / dim_a, type_as=K)
    v = nx.full((dim_b,), 1. / dim_b, type_as=K)
    G = u[:, None] * K * v[None, :]

    viol = nx.sum(G, axis=1) - a # compresses all the columns into (by adding all the values in each row into one column vector 
    viol_2 = nx.sum(G, axis=0) - b #compress all the rows by summing all the valeus across each column into one  row vector
    stopThr_val = 1
    if log:
        log = dict()
        log['u'] = u
        log['v'] = v

    for ii in range(numItermax):
        i_1 = int(random.randrange(0, len(viol))) 
        i_2 = int(random.randrange(0, len(viol_2))) 

        m_viol_1 = nx.abs(viol[i_1])
        m_viol_2 = nx.abs(viol_2[i_2])
        stopThr_val = nx.maximum(m_viol_1, m_viol_2)

        if m_viol_1 > m_viol_2:
            old_u = u[i_1]
            new_u = a[i_1] / nx.dot(K[i_1, :], v)
            G[i_1, :] = new_u * K[i_1, :] * v

            viol[i_1] = nx.dot(new_u * K[i_1, :], v) - a[i_1]
            viol_2 += (K[i_1, :].T * (new_u - old_u) * v)
            u[i_1] = new_u
        else:
            old_v = v[i_2]
            new_v = b[i_2] / nx.dot(K[:, i_2].T, u)
            G[:, i_2] = u * K[:, i_2] * new_v
            # aviol = (G@one_m - a)
            # aviol_2 = (G.T@one_n - b)
            viol += (-old_v + new_v) * K[:, i_2] * u
            viol_2[i_2] = new_v * nx.dot(K[:, i_2], u) - b[i_2]
            v[i_2] = new_v

        if stopThr_val <= stopThr:
            break
    else:
        if warn:
            warnings.warn("Sinkhorn did not converge. You might want to "
                          "increase the number of iterations `numItermax` "
                          "or the regularization parameter `reg`.")

    if log:
        log["n_iter"] = ii
        log['u'] = u
        log['v'] = v

    if log:
        return G, log
    else:
        return G