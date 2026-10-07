"""One case of benchmark.py: fits naffprop or scikit-learn once and prints, as
JSON, the time, the extra peak memory, the number of clusters and of
iterations. benchmark.py runs it in a fresh process for each case, so that the
peak memory (max RSS) of one case does not hide the next one. It can also be
run on its own:

    uv run python benchmark_case.py naffprop 2000 0.9 1000 100

Arguments: library (naffprop or sklearn), n, damping, max_iter, convergence_iter.
"""

import json
import resource
import sys
import time

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import euclidean_distances


def similarities(n):
    """-||x_i - x_j||^2 for n blob points, computed by blocks of rows: no
    n x n temporary inflates the peak memory before the fit."""
    x, _ = make_blobs(n_samples=n, centers=10, cluster_std=1.0, random_state=0)
    s = np.empty((n, n))
    for i in range(0, n, 100):
        s[i:i + 100] = -euclidean_distances(x[i:i + 100], x, squared=True)
    return s


def max_rss():
    """Peak memory of this process in bytes (ru_maxrss: KiB on Linux, bytes
    on macOS; not available on Windows)."""
    scale = 1 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale


def main(lib, n, damping, max_iter, conv):
    s = similarities(n)
    p = float(np.median(s[:100, 100:]))  # median of a sample: no n x n temporary
    if lib == "naffprop":
        from naffprop import apcluster

        def fit():
            res = apcluster(s, p=p, lam=damping, maxits=max_iter, convits=conv, seed=0)
            return len(res.exemplars), res.iterations
    elif lib == "sklearn":
        from sklearn.cluster import affinity_propagation

        def fit():
            centers, _, its = affinity_propagation(
                s, preference=p, damping=damping, max_iter=max_iter,
                convergence_iter=conv, random_state=0, return_n_iter=True)
            return len(centers), its
    else:
        raise SystemExit(f"unknown library {lib!r}: naffprop or sklearn")
    before = max_rss()
    t = time.perf_counter()
    k, its = fit()
    t = time.perf_counter() - t
    mem = (max_rss() - before) / 2**20
    print(json.dumps(dict(time=t, mem=mem, k=k, its=its)))


if __name__ == "__main__":
    lib, n, damping, max_iter, conv = sys.argv[1:6]
    main(lib, int(n), float(damping), int(max_iter), int(conv))
