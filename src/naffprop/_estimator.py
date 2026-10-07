"""A drop-in replacement for sklearn.cluster.AffinityPropagation."""

import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import euclidean_distances
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data

import naffprop


class AffinityPropagation(ClusterMixin, BaseEstimator):
    """Affinity propagation clustering, computed in Nim, with the parameters
    of sklearn.cluster.AffinityPropagation and what R's apcluster adds.

    Differences from scikit-learn:

    - R's defaults: damping=0.9, max_iter=1000, convergence_iter=100
      (scikit-learn: 0.5, 200 and 15, which often stop before convergence).
    - The default preference is the median of the similarities off the
      diagonal, as in R (scikit-learn includes the diagonal).
    - preference_quantile: another quantile than the median (R's q).
    - n_clusters: search the preference that gives that many clusters
      (R's apclusterK).
    - If it does not converge, the clustering found is kept (with a
      ConvergenceWarning), as in R, instead of labelling every point -1.

    Parameters
    ----------
    damping : float in [0.5, 1), default 0.9
    max_iter : int, default 1000
    convergence_iter : int, default 100
        Stop when the exemplars have not changed for that many iterations.
    copy : bool, default True
        Kept for compatibility: the input is never modified.
    preference : float or array of shape (n_samples,), default None
        Larger preferences give more clusters. None: the quantile
        preference_quantile of the similarities.
    preference_quantile : float in [0, 1], default None (the median)
    n_clusters : int, default None
        Number of clusters to look for (preference and preference_quantile
        are then ignored).
    affinity : {"euclidean", "precomputed"}, default "euclidean"
        "euclidean": negative squared Euclidean distance, as scikit-learn.
        "precomputed": a similarity matrix, where -Inf means never link
        that pair (as in R).
    verbose : bool, default False
    random_state : int, RandomState instance or None, default None
        Seed of the tiny noise added to the similarities to break ties.

    Attributes
    ----------
    cluster_centers_indices_, cluster_centers_ (not for "precomputed"),
    labels_, affinity_matrix_, n_iter_, n_features_in_, as scikit-learn;
    preference_ (the preference of each point) and netsim_ (the net
    similarity, the objective of affinity propagation).
    """

    def __init__(self, *, damping=0.9, max_iter=1000, convergence_iter=100,
                 copy=True, preference=None, preference_quantile=None,
                 n_clusters=None, affinity="euclidean", verbose=False,
                 random_state=None):
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.copy = copy
        self.preference = preference
        self.preference_quantile = preference_quantile
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.verbose = verbose
        self.random_state = random_state

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.pairwise = self.affinity == "precomputed"
        return tags

    def fit(self, X, y=None):
        """Fit the clustering from features, or from a similarity matrix if
        affinity="precomputed"."""
        if self.affinity not in ("euclidean", "precomputed"):
            raise ValueError(f"affinity must be 'euclidean' or 'precomputed', got {self.affinity!r}")
        precomputed = self.affinity == "precomputed"
        # -Inf similarities are allowed: that pair is never linked.
        X = validate_data(self, X, dtype=np.float64, copy=False,
                          ensure_all_finite=not precomputed)
        if precomputed:
            if X.shape[0] != X.shape[1]:
                raise ValueError(f"precomputed affinity must be square, got {X.shape}")
            if np.isnan(X).any() or np.isposinf(X).any():
                raise ValueError("precomputed affinity must not contain NaN or +Inf")
            S = X
        else:
            S = -euclidean_distances(X, squared=True)
        self.affinity_matrix_ = S
        n = S.shape[0]
        seed = check_random_state(self.random_state).randint(np.iinfo(np.int32).max)
        kwargs = dict(maxits=self.max_iter, convits=self.convergence_iter,
                      lam=self.damping, seed=seed)
        if n == 1:
            res = naffprop.apcluster(S, p=0.0, **kwargs)
        elif self.n_clusters is not None:
            k = self.n_clusters
            if not 1 <= k <= n:
                raise ValueError(f"n_clusters must be between 1 and {n}, got {k}")
            if k == n:
                # Every point its own exemplar: the largest meaningful preference.
                res = naffprop.apcluster(S, p=naffprop.preference_range(S)[1] + 1, **kwargs)
            elif k == 1:
                # Well below the preference where a second cluster pays off.
                lo, hi = naffprop.preference_range(S)
                res = naffprop.apcluster(S, p=lo - (hi - lo), **kwargs)
            else:
                res = naffprop.apcluster_k(S, k, prc=0, **kwargs)
        else:
            res = naffprop.apcluster(S, p=self.preference, q=self.preference_quantile, **kwargs)
        if self.verbose:
            print(f"{len(res)} clusters after {res.iterations} iterations")
        if not res.converged:
            warnings.warn("Affinity propagation did not converge: increase max_iter "
                          "or convergence_iter, or the damping.", ConvergenceWarning)
        self.cluster_centers_indices_ = res.exemplars
        self.labels_ = res.labels
        self.n_iter_ = res.iterations
        self.preference_ = res.p
        self.netsim_ = res.netsim
        if self.affinity != "precomputed":
            self.cluster_centers_ = X[res.exemplars].copy()
        return self

    def predict(self, X):
        """The cluster of each sample: the one of its nearest exemplar."""
        check_is_fitted(self)
        X = validate_data(self, X, dtype=np.float64, reset=False)
        if self.affinity == "precomputed":
            raise ValueError("predict is not supported with affinity='precomputed'")
        if len(self.cluster_centers_) == 0:
            warnings.warn("No exemplars were found: every sample is labelled -1.",
                          ConvergenceWarning)
            return np.full(X.shape[0], -1, dtype=np.intp)
        d = euclidean_distances(X, self.cluster_centers_, squared=True)
        return d.argmin(axis=1).astype(np.intp)

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_
