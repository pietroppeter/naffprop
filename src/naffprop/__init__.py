"""Affinity propagation in Nim, for Python: R's apcluster features behind a
scikit-learn API."""

from dataclasses import dataclass

import numpy as np

from naffprop import core as _core

__all__ = [
    "AffinityPropagation",
    "APResult",
    "apcluster",
    "apcluster_k",
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

    @property
    def clusters(self):
        """For each exemplar, the indices of the points in its cluster."""
        return [np.flatnonzero(self.labels == k) for k in range(len(self.exemplars))]

    def __len__(self):
        return len(self.exemplars)


def _similarities(s):
    s = np.asarray(s, dtype=np.float64)
    if s.ndim != 2 or s.shape[0] != s.shape[1] or s.shape[0] == 0:
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


def neg_dist_mat(x, r=1):
    """Negative Euclidean distances between the rows of `x`, to the power `r`,
    as R's negDistMat(x, r). r=2 gives scikit-learn's default similarity."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError("x must be a 1D or 2D array")
    if r <= 0:
        raise ValueError(f"r must be positive, got {r}")
    return _core.negDistMat(x, float(r))


def preference_range(s, exact=False):
    """(pmin, pmax): the preferences that give from 1 or 2 clusters to one
    cluster per point, as R's preferenceRange. Without `exact`, pmin is a
    cheaper (O(n^2) instead of O(n^3)) lower bound."""
    s = _similarities(s)
    if len(s) < 2:
        raise ValueError("need at least 2 points")
    return _core.preferenceRange(s, bool(exact))


def apcluster(s, p=None, q=None, maxits=1000, convits=100, lam=0.9,
              nonoise=False, seed=None):
    """Affinity propagation on the similarity matrix `s`, as R's apcluster.

    p: preference, one value for all points or one per point. If None, the
       quantile `q` (default 0.5, the median) of the off-diagonal similarities.
    lam: damping factor in [0.5, 1).
    maxits, convits: stop after maxits iterations, or when the exemplars have
       not changed for convits iterations.
    nonoise: don't add the tiny noise that removes degeneracies (ties).
    seed: seed of that noise.
    """
    s = _similarities(s)
    n = len(s)
    if not 0.5 <= lam < 1:
        raise ValueError(f"lam must be in [0.5, 1), got {lam}")
    if maxits < 1 or convits < 1:
        raise ValueError("maxits and convits must be positive")
    if p is None:
        if q is not None and not 0 <= q <= 1:
            raise ValueError(f"q must be in [0, 1], got {q}")
        v = _off_diagonal(s)
        p = np.quantile(v, 0.5 if q is None else q) if len(v) else 0.0
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    if len(p) not in (1, n):
        raise ValueError(f"p must be one value or one per point ({n}), got {len(p)}")
    if not np.isfinite(p).all():
        raise ValueError("p must be finite")
    ex, labels, its, conv, netsim, dpsim, expref = _core.apcluster(
        s, p, int(maxits), int(convits), float(lam), not nonoise, _seed(seed))
    return APResult(
        exemplars=np.asarray(ex, dtype=np.intp),
        labels=np.asarray(labels, dtype=np.intp),
        p=np.broadcast_to(p, n).copy(),
        iterations=its,
        converged=conv,
        netsim=netsim,
        dpsim=dpsim,
        expref=expref,
    )


def apcluster_k(s, k, prc=10, bimaxit=20, exact=False, maxits=1000,
                convits=100, lam=0.9, nonoise=False, seed=None):
    """Affinity propagation with (about) `k` clusters, as R's apclusterK: a
    bisection on the preference, within preference_range(s), that stops when
    the number of clusters is within `prc` percent of k, or after `bimaxit`
    steps (use prc=0 to ask for exactly k)."""
    s = _similarities(s)
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
