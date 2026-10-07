"""One case of benchmark.py: fits naffprop or scikit-learn once and prints, as
JSON, the time, the extra peak memory, the number of clusters and of
iterations. benchmark.py runs it in a fresh process for each case, so that the
peak memory (max RSS) of one case does not hide the next one. It can also be
run on its own:

    uv run python benchmark_case.py naffprop 2000 0.9 1000 100

Arguments: library, n, damping, max_iter, convergence_iter. The library is
sklearn, or naffprop with its options: naffprop (float64), naffprop-f32
(float32), naffprop-f32-inplace (float32 with copy=False: no copy of the
similarity matrix), naffprop-sparse (the 10% nearest neighbours of each point
only, as a scipy.sparse matrix), naffprop-leveraged (apcluster_l on 10% of the
points, 5 sweeps: the n x n similarity matrix is never built, so its memory
includes its similarities).
"""

import json
import resource
import sys
import time

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import euclidean_distances


def points(n):
    return make_blobs(n_samples=n, centers=10, cluster_std=1.0, random_state=0)[0]


def similarities(x, dtype=np.float64):
    """-||x_i - x_j||^2, computed by blocks of rows: no n x n temporary
    inflates the peak memory before the fit."""
    n = len(x)
    s = np.empty((n, n), dtype=dtype)
    for i in range(0, n, 100):
        s[i:i + 100] = -euclidean_distances(x[i:i + 100], x, squared=True)
    return s


def nearest_neighbours(x, k):
    """-||x_i - x_j||^2 for the k nearest neighbours j of each point i, as a
    sparse matrix, computed by blocks of rows."""
    from scipy.sparse import csr_array

    n = len(x)
    rows, cols, vals = [], [], []
    for i in range(0, n, 100):
        d = euclidean_distances(x[i:i + 100], x, squared=True)
        d[np.arange(len(d)), np.arange(i, i + len(d))] = np.inf
        nb = np.argpartition(d, k, axis=1)[:, :k]
        rows.append(np.repeat(np.arange(i, i + len(d)), k))
        cols.append(nb.ravel())
        vals.append(-np.take_along_axis(d, nb, axis=1).ravel())
    return csr_array((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                     shape=(n, n))


def max_rss():
    """Peak memory of this process in bytes (ru_maxrss: KiB on Linux, bytes
    on macOS; not available on Windows)."""
    scale = 1 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale


def main(lib, n, damping, max_iter, conv):
    x = points(n)
    # The median of a sample of the similarities: no n x n temporary.
    p = float(np.median(-euclidean_distances(x[:100], x[100:], squared=True)))
    if lib == "naffprop-sparse":
        s = nearest_neighbours(x, n // 10)
    elif lib != "naffprop-leveraged":
        s = similarities(x, np.float32 if "-f32" in lib else np.float64)
    if lib in ("naffprop", "naffprop-f32", "naffprop-f32-inplace", "naffprop-sparse"):
        from naffprop import apcluster

        def fit():
            res = apcluster(s, p=p, lam=damping, maxits=max_iter, convits=conv, seed=0,
                            copy=not lib.endswith("-inplace"))
            return res.labels, res.iterations
    elif lib == "naffprop-leveraged":
        from naffprop import apcluster_l

        def fit():
            res = apcluster_l(x, frac=0.1, sweeps=5, p=p, lam=damping, maxits=max_iter,
                              convits=conv, seed=0)
            return res.labels, res.iterations
    elif lib == "sklearn":
        from sklearn.cluster import affinity_propagation

        def fit():
            _, labels, its = affinity_propagation(
                s, preference=p, damping=damping, max_iter=max_iter,
                convergence_iter=conv, random_state=0, return_n_iter=True)
            return labels, its
    else:
        raise SystemExit(f"unknown library {lib!r}: naffprop or sklearn")
    before = max_rss()
    t = time.perf_counter()
    labels, its = fit()
    t = time.perf_counter() - t
    mem = (max_rss() - before) / 2**20
    k = len(set(labels.tolist()) - {-1})
    print(json.dumps(dict(time=t, mem=mem, k=k, its=its, labels=labels.tolist())))


if __name__ == "__main__":
    lib, n, damping, max_iter, conv = sys.argv[1:6]
    main(lib, int(n), float(damping), int(max_iter), int(conv))
