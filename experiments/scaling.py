"""The options for large n against the full dense run, by net similarity (the
objective of affinity propagation, higher is better) with the same preference:
scaleap=True, sparse similarities (the 10% nearest neighbours of each point)
and leveraged AP (10% of the points, 5 sweeps). Data and preference as in
benchmark.py.

    uv run python experiments/scaling.py
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, euclidean_distances

sys.path.insert(0, str(Path(__file__).parents[1]))
from benchmark_case import nearest_neighbours, points, similarities  # noqa: E402

from naffprop import apcluster, apcluster_l  # noqa: E402


def main():
    print("| n | option | clusters | net similarity | below the full run | ARI vs full |")
    print("|--:|:--|--:|--:|--:|--:|")
    for n in (4000, 8000):
        x = points(n)
        p = float(np.median(-euclidean_distances(x[:100], x[100:], squared=True)))
        full = apcluster(similarities(x), p=p, seed=0, copy=False)
        runs = [
            ("full", full),
            ("scaleap", apcluster(similarities(x), p=p, seed=0, copy=False, scaleap=True)),
            ("sparse, 10% neighbours", apcluster(nearest_neighbours(x, n // 10), p=p, seed=0)),
            ("leveraged, 10% sample", apcluster_l(x, frac=0.1, sweeps=5, p=p, seed=0)),
        ]
        for name, r in runs:
            gap = 100 * (full.netsim - r.netsim) / abs(full.netsim)
            gap = 0.0 if abs(gap) < 0.05 else gap
            print(f"| {n:,} | {name} | {len(r.exemplars)} | {r.netsim:.1f} | {gap:.1f}% "
                  f"| {adjusted_rand_score(full.labels, r.labels):.2f} |", flush=True)


if __name__ == "__main__":
    main()
