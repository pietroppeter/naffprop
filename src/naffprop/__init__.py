"""Affinity propagation in Nim, for Python: R's apcluster features behind a
scikit-learn API."""

import math
from dataclasses import dataclass

import numpy as np

from naffprop import core as _core

__all__ = [
    "AffinityPropagation",
    "APResult",
    "apcluster",
    "apcluster_k",
    "apcluster_l",
    "neg_dist_mat",
    "preference_range",
]


@dataclass
class APResult:
    """Outcome of affinity propagation, as R's APResult."""

    exemplars: np.ndarray  # indices of the exemplars, increasing
    labels: np.ndarray  # position in `exemplars` of each point's exemplar (-1: none)
    p: np.ndarray  # the preference of each point
    iterations: int
    converged: bool
    netsim: float  # dpsim + expref, the objective AP maximizes
    dpsim: float  # sum of similarities of points to their exemplars
    expref: float  # sum of the preferences of the exemplars
    sel: np.ndarray = None  # apcluster_l: the sampled points of the best sweep

    @property
    def clusters(self):
        """For each exemplar, the indices of the points in its cluster."""
        return [np.flatnonzero(self.labels == k) for k in range(len(self.exemplars))]

    def __len__(self):
        return len(self.exemplars)


def _dtype(s, dtype):
    """float32 or float64: `dtype` if given, else float32 for a float32 `s`."""
    if dtype is None:
        dtype = np.float32 if getattr(s, "dtype", None) == np.float32 else np.float64
    dtype = np.dtype(dtype)
    if dtype not in (np.float32, np.float64):
        raise ValueError(f"dtype must be float32 or float64, got {dtype}")
    return dtype


def _similarities(s, dtype=None, copy=False, square=True):
    """`s` as a C-contiguous float32 or float64 array: a copy if `copy`, else
    `s` itself when it already is one."""
    if _issparse(s):
        raise TypeError("sparse similarities are only supported by apcluster")
    dtype = _dtype(s, dtype)
    s = np.array(s, dtype=dtype, order="C", copy=True if copy else None)
    if s.ndim != 2 or (square and s.shape[0] != s.shape[1]) or s.shape[0] == 0:
        raise ValueError(f"s must be a non-empty square matrix, got shape {s.shape}")
    if np.isposinf(s).any():
        raise ValueError("+Inf similarities: use a large finite value instead")
    return s


def _off_diagonal(s):
    """The finite similarities off the diagonal, as R uses to choose p."""
    v = s[~np.eye(len(s), dtype=bool)]
    return v[v > -np.inf]


def _seed(seed):
    if seed is None:
        return int(np.random.default_rng().integers(2**62))
    return int(seed)


def neg_dist_mat(x, r=1, sel=None, dtype=np.float64):
    """Negative Euclidean distances between the rows of `x`, to the power `r`,
    as R's negDistMat(x, r). r=2 gives scikit-learn's default similarity.
    sel: only the distances to these rows (an n x len(sel) matrix), as R's
    negDistMat(x, sel, r), for apcluster_l.
    dtype: float64, or float32 for half the memory."""
    dtype = _dtype(None, dtype)
    x = np.ascontiguousarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError("x must be a 1D or 2D array")
    if r <= 0:
        raise ValueError(f"r must be positive, got {r}")
    sel = np.arange(len(x)) if sel is None else _selection(sel, len(x))
    return _core.negDistMat(x, float(r), sel.tolist(), dtype == np.float32)


def _selection(sel, n):
    sel = np.asarray(sel, dtype=np.intp).reshape(-1)
    if len(sel) == 0 or sel.min() < 0 or sel.max() >= n or (np.diff(sel) <= 0).any():
        raise ValueError(f"sel must be increasing indices between 0 and {n - 1}")
    return sel


def preference_range(s, exact=False):
    """(pmin, pmax): the preferences that give from 1 or 2 clusters to one
    cluster per point, as R's preferenceRange. Without `exact`, pmin is a
    cheaper (O(n^2) instead of O(n^3)) lower bound."""
    s = _similarities(s)
    if len(s) < 2:
        raise ValueError("need at least 2 points")
    return _core.preferenceRange(s, bool(exact))


def _check_params(lam, maxits, convits, q):
    if not 0.5 <= lam < 1:
        raise ValueError(f"lam must be in [0.5, 1), got {lam}")
    if maxits < 1 or convits < 1:
        raise ValueError("maxits and convits must be positive")
    if q is not None and not 0 <= q <= 1:
        raise ValueError(f"q must be in [0, 1], got {q}")


def _preferences(p, q, similarities, n):
    """p as one value per point: if None, the quantile q (default: the median)
    of `similarities` (the finite ones off the diagonal)."""
    if p is None:
        p = np.quantile(similarities, 0.5 if q is None else q) if len(similarities) else 0.0
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    if len(p) not in (1, n):
        raise ValueError(f"p must be one value or one per point ({n}), got {len(p)}")
    if not np.isfinite(p).all():
        raise ValueError("p must be finite")
    return np.broadcast_to(p, n).copy()


def _result(out, p, sel=None):
    ex, labels, its, conv, netsim, dpsim, expref = out
    return APResult(
        exemplars=np.asarray(ex, dtype=np.intp),
        labels=np.asarray(labels, dtype=np.intp),
        p=p,
        iterations=its,
        converged=conv,
        netsim=netsim,
        dpsim=dpsim,
        expref=expref,
        sel=sel,
    )


def apcluster(s, p=None, q=None, maxits=1000, convits=100, lam=0.9,
              nonoise=False, seed=None, dtype=None, copy=True):
    """Affinity propagation on the similarity matrix `s`, as R's apcluster.

    s: a square array, or a scipy.sparse matrix: then messages are passed only
       between the pairs it stores, so memory grows with their number, and
       the pairs not stored (or -Inf) are never linked. Its diagonal is
       ignored (the preferences take its place).
    p: preference, one value for all points or one per point. If None, the
       quantile `q` (default 0.5, the median) of the off-diagonal similarities
       (for a sparse s, of the stored ones).
    lam: damping factor in [0.5, 1).
    maxits, convits: stop after maxits iterations, or when the exemplars have
       not changed for convits iterations.
    nonoise: don't add the tiny noise that removes degeneracies (ties).
    seed: seed of that noise.
    dtype: float64 or float32, in which the similarities and the messages are
       stored: float32 halves the memory. None: float32 if s is float32.
    copy: if False, and s is a C-contiguous array of that dtype, work in s
       itself instead of a copy (saves one n x n matrix): s then holds the
       noise and the preferences on its diagonal.
    """
    _check_params(lam, maxits, convits, q)
    if _issparse(s):
        return _apcluster_sparse(s, p, q, maxits, convits, lam, nonoise, seed, dtype)
    s = _similarities(s, dtype, copy)
    p = _preferences(p, q, _off_diagonal(s), len(s))
    return _result(_core.apcluster(s, p.tolist(), int(maxits), int(convits), float(lam),
                                   not nonoise, _seed(seed)), p)


def _issparse(s):
    try:
        from scipy import sparse
    except ImportError:  # pragma: no cover - scipy comes with scikit-learn
        return False
    return sparse.issparse(s)


def _apcluster_sparse(s, p, q, maxits, convits, lam, nonoise, seed, dtype):
    from scipy import sparse

    dtype = _dtype(s, dtype)
    s = sparse.coo_array(s)
    n = s.shape[0]
    if s.ndim != 2 or s.shape[1] != n or n == 0:
        raise ValueError(f"s must be a non-empty square matrix, got shape {s.shape}")
    # The stored pairs off the diagonal, without -Inf (never linked).
    keep = (s.row != s.col) & ~np.isneginf(s.data)
    row, col, val = s.row[keep], s.col[keep], s.data[keep].astype(np.float64)
    if np.isnan(val).any() or np.isposinf(val).any():
        raise ValueError("s must not contain NaN or +Inf")
    p = _preferences(p, q, val, n)
    # By row, sorted by column, with the preferences on the diagonal.
    diag = np.arange(n)
    m = sparse.csr_array((np.concatenate([val, p]).astype(dtype),
                          (np.concatenate([row, diag]), np.concatenate([col, diag]))),
                         shape=(n, n))
    m.sum_duplicates()
    m.sort_indices()
    rows = np.repeat(diag, np.diff(m.indptr))
    out = _core.apclusterSparse(
        n, m.indptr.astype(np.int64), m.indices.astype(np.int64),
        np.flatnonzero(m.indices == rows).astype(np.int64), np.ascontiguousarray(m.data),
        int(maxits), int(convits), float(lam), not nonoise, _seed(seed))
    return _result(out, p)


def apcluster_l(x, frac, sweeps, s=None, p=None, q=None, maxits=1000,
                convits=100, lam=0.9, nonoise=False, seed=None, dtype=np.float64,
                sel=None):
    """Leveraged affinity propagation, as R's apclusterL: affinity propagation
    on the similarities of all the points to a random sample of them, a
    fraction `frac`, which are the only possible exemplars. Memory and time
    grow with n * frac * n instead of n^2, and the n x n similarity matrix is
    never built.

    It runs `sweeps` times: each sweep samples the exemplars of the best
    clustering so far plus random other points, and the clustering with the
    largest net similarity is returned (its sample in `sel`).

    x: the data, passed to `s`.
    s: s(x, sel) gives the similarities of every point to the points `sel`,
       as an n x len(sel) array (overwritten with the noise). Default:
       neg_dist_mat(x, r=2, sel=sel), as scikit-learn's similarity.
    p, q: the preference, as apcluster (q of the similarities computed, except
       those of the sampled points to themselves).
    dtype: float64 or float32 (half the memory), for the default s.
    sel: the sample of the first sweep (increasing indices), instead of a
       random one; frac is then len(sel) / n.
    Other arguments as apcluster.
    """
    _check_params(lam, maxits, convits, q)
    if sel is None and not (frac is not None and 0 < frac <= 1):
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    if sweeps < 1:
        raise ValueError(f"sweeps must be positive, got {sweeps}")
    n = len(x)
    if n < 2:
        raise ValueError("need at least 2 points")
    if s is None:
        def s(x, sel):
            return neg_dist_mat(x, 2, sel=sel, dtype=dtype)
    rng = np.random.default_rng(seed)
    if sel is None:
        nsel = max(math.ceil(n * frac), 2)
        sel = np.sort(rng.choice(n, nsel, replace=False))
    else:
        sel = _selection(sel, n)
        nsel = len(sel)
    best = None
    for _ in range(sweeps):
        sim = s(x, sel)
        if getattr(sim, "shape", None) != (n, len(sel)):
            raise ValueError(f"s(x, sel) must be an array of shape {(n, len(sel))}")
        sim = _similarities(sim, None, False, square=False)
        self_pairs = np.zeros(sim.shape, dtype=bool)
        self_pairs[sel, np.arange(len(sel))] = True
        v = sim[~self_pairs] if p is None else np.empty(0)
        pp = _preferences(p, q, v[v > -np.inf], n)
        out = _core.apclusterLeveraged(sim, sel.tolist(), pp.tolist(), int(maxits), int(convits),
                                       float(lam), not nonoise, int(rng.integers(2**62)))
        res = _result(out, pp, sel)
        if best is None or res.netsim > best.netsim or (
                math.isnan(best.netsim) and not math.isnan(res.netsim)):
            best = res
        # Next sample: the exemplars so far, and random other points.
        keep = best.exemplars
        others = np.setdiff1d(np.arange(n), keep)
        sel = np.sort(np.concatenate([keep, rng.choice(others, nsel - len(keep), replace=False)]))
    return best


def apcluster_k(s, k, prc=10, bimaxit=20, exact=False, maxits=1000,
                convits=100, lam=0.9, nonoise=False, seed=None, dtype=None):
    """Affinity propagation with (about) `k` clusters, as R's apclusterK: a
    bisection on the preference, within preference_range(s), that stops when
    the number of clusters is within `prc` percent of k, or after `bimaxit`
    steps (use prc=0 to ask for exactly k)."""
    s = _similarities(s, dtype)
    n = len(s)
    if not 2 <= k < n:
        raise ValueError(f"k must be between 2 and {n - 1}, got {k}")
    lo, hi = preference_range(s, exact)
    if not np.isfinite(lo) or lo >= hi:
        raise ValueError("could not find a valid preference range for s")
    # The same seed in every run: the same noise, so the bisection is
    # deterministic (R adds the noise once, before bisecting).
    seed = _seed(seed)

    def run(p):
        return apcluster(s, p, maxits=maxits, convits=convits, lam=lam,
                         nonoise=nonoise, seed=seed)

    def off(res):
        return abs(len(res) - k) * 100 / k

    # A better lower bound than pmin, close to pmax, before bisecting.
    for e in (-3, -2, -1):
        p = hi - 10**e * (hi - lo)
        res = run(p)
        if 0 < len(res) < k:
            lo = p
            break
    steps = 0
    while off(res) > prc and steps < bimaxit:
        steps += 1
        p = (lo + hi) / 2
        res = run(p)
        if len(res) < k:
            lo = p
        else:
            hi = p
    if off(res) > prc:
        import warnings
        warnings.warn(f"found {len(res)} clusters instead of {k}: increase bimaxit",
                      stacklevel=2)
    return res


from naffprop._estimator import AffinityPropagation  # noqa: E402
