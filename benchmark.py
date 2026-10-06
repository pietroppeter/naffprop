# /// script
# requires-python = ">=3.9"
# dependencies = ["naffprop", "numpy", "scikit-learn"]
#
# [tool.uv.sources]
# naffprop = { path = "." }
# ///
"""Time naffprop against scikit-learn's AffinityPropagation, on the same
similarity matrix with the same parameters, and measure the memory each
needs on top of that matrix.

    uv run benchmark.py

Every case runs in its own process, so the peak memory (max RSS) of one does
not hide the next one.
"""

import json
import subprocess
import sys

CASE = r"""
import json, resource, sys, time
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import euclidean_distances

lib, n, damping, max_iter, conv = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
x, _ = make_blobs(n_samples=n, centers=10, cluster_std=1.0, random_state=0)
s = np.empty((n, n))
for i in range(0, n, 100):  # by blocks, so that no n x n temporary inflates the baseline
    s[i:i + 100] = -euclidean_distances(x[i:i + 100], x, squared=True)
p = float(np.median(s[:100, 100:]))  # median of a sample: no n x n temporary
if lib == "naffprop":
    from naffprop import apcluster
    fit = lambda: apcluster(s, p=p, lam=damping, maxits=max_iter, convits=conv, seed=0)
else:
    from sklearn.cluster import affinity_propagation
    fit = lambda: affinity_propagation(s, preference=p, damping=damping, max_iter=max_iter,
                                       convergence_iter=conv, random_state=0, return_n_iter=True)
scale = 1024 if sys.platform != "darwin" else 1  # ru_maxrss: KiB on Linux, bytes on macOS
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
t = time.perf_counter()
res = fit()
t = time.perf_counter() - t
after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
if lib == "naffprop":
    k, its = len(res.exemplars), res.iterations
else:
    k, its = len(res[0]), res[2]
print(json.dumps(dict(time=t, mem=(after - before) / 2**20, k=k, its=its)))
"""


def run(lib, n, damping, max_iter, conv):
    out = subprocess.run([sys.executable, "-c", CASE, lib, str(n), str(damping), str(max_iter), str(conv)],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out.strip().splitlines()[-1])


SIZES = [500, 1_000, 2_000, 4_000]
SETTINGS = [("sklearn defaults", 0.5, 200, 15), ("R defaults", 0.9, 1000, 100)]

print("Time (s) and extra peak memory (MiB, on top of the n x n similarity matrix).\n")
print("| n | parameters | naffprop | scikit-learn | speedup | naffprop memory | scikit-learn memory | clusters | iterations |")
print("|--:|:-----------|---------:|-------------:|--------:|----------------:|--------------------:|---------:|-----------:|")
for n in SIZES:
    for name, damping, max_iter, conv in SETTINGS:
        a = run("naffprop", n, damping, max_iter, conv)
        b = run("sklearn", n, damping, max_iter, conv)
        print(f"| {n:,} | {name} | {a['time']:.2f} | {b['time']:.2f} | {b['time'] / a['time']:.1f}x "
              f"| {a['mem']:.0f} | {b['mem']:.0f} | {a['k']} / {b['k']} | {a['its']} / {b['its']} |",
              flush=True)
