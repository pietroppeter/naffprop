"""Affinity propagation in numpy, line by line from Frey and Dueck's MATLAB
code (apcluster.m, without noise): the reference naffprop is checked against."""

import numpy as np


def reference_apcluster(s, p, maxits=1000, convits=100, lam=0.9):
    s = np.array(s, dtype=float)
    n = len(s)
    s[np.arange(n), np.arange(n)] = p
    a = np.zeros((n, n))
    r = np.zeros((n, n))
    e = np.zeros((n, convits), dtype=bool)
    rows = np.arange(n)
    for i in range(maxits):
        # responsibilities
        as_ = a + s
        y = as_.argmax(axis=1)
        m1 = as_[rows, y]
        as_[rows, y] = -np.inf
        m2 = as_.max(axis=1)
        rnew = s - m1[:, None]
        rnew[rows, y] = s[rows, y] - m2
        r = (1 - lam) * rnew + lam * r
        # availabilities
        rp = np.maximum(r, 0)
        rp[rows, rows] = r[rows, rows]
        anew = rp.sum(axis=0)[None, :] - rp
        dA = anew[rows, rows].copy()
        anew = np.minimum(anew, 0)
        anew[rows, rows] = dA
        a = (1 - lam) * anew + lam * a
        # convergence
        E = (np.diag(a) + np.diag(r)) > 0
        e[:, i % convits] = E
        K = E.sum()
        if i >= convits - 1 or i >= maxits - 1:
            se = e.sum(axis=1)
            unconverged = ((se == convits) | (se == 0)).sum() != n
            if (not unconverged and K > 0) or i == maxits - 1:
                break
    I = np.flatnonzero(np.diag(a) + np.diag(r) > 0)
    K = len(I)
    if K == 0:
        return np.array([], dtype=int), np.full(n, -1), i + 1
    c = s[:, I].argmax(axis=1)
    c[I] = np.arange(K)
    for k in range(K):
        ii = np.flatnonzero(c == k)
        I[k] = ii[s[np.ix_(ii, ii)].sum(axis=0).argmax()]
    c = s[:, I].argmax(axis=1)
    c[I] = np.arange(K)
    idx = I[c]
    exemplars = np.unique(idx)
    labels = np.searchsorted(exemplars, idx)
    return exemplars, labels, i + 1
