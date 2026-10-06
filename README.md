# naffprop

[Affinity propagation] clustering written in [Nim] for Python. It aims to be the best of R's
[apcluster] behind scikit-learn's API, and it is built with
[nimlang](https://github.com/pietroppeter/uv-add-nimlang).

> AI disclosure: this project is mostly vibed. Currently [level 7](https://www.visidata.org/blog/2026/ai/#level-6%3A-bots-coded%2C-human-understands-mostly) on visidata AI scale: Human specced, bots coded.

```python
from naffprop import AffinityPropagation

ap = AffinityPropagation().fit(X)              # drop-in for sklearn.cluster.AffinityPropagation
ap.labels_, ap.cluster_centers_indices_

AffinityPropagation(preference_quantile=0.1)   # fewer clusters: R's q
AffinityPropagation(n_clusters=5)              # exactly 5 clusters: R's apclusterK
AffinityPropagation(affinity="precomputed").fit(S)   # your own similarity matrix
```

The same features are also available as functions named after R's:

```python
from naffprop import apcluster, apcluster_k, neg_dist_mat, preference_range

S = neg_dist_mat(X, r=2)        # -||x_i - x_j||^2, as R's negDistMat
res = apcluster(S, q=0.5)       # APResult: exemplars, labels, clusters, netsim, iterations, converged
res = apcluster_k(S, 5)
pmin, pmax = preference_range(S)
```

## Compared with scikit-learn and R

Affinity propagation is the same algorithm in all three: Frey and Dueck's MATLAB code, with
the same message updates, damping, convergence test and refinement of the exemplars. With the
same similarities, preference and parameters, naffprop finds the same exemplars, on the test data, as
scikit-learn and as a line-by-line numpy port of the MATLAB code (both are checked in the
tests). The differences are in what is offered around it:

| | scikit-learn | R apcluster | naffprop |
|:--|:--|:--|:--|
| defaults (damping, max iterations, convergence) | 0.5, 200, 15 | 0.9, 1000, 100 | R's |
| default preference: median of the similarities | including the diagonal | off the diagonal | R's |
| preference as a quantile | no | `q` | `preference_quantile`, `q` |
| ask for k clusters | no | `apclusterK` | `n_clusters`, `apcluster_k` |
| preference range | no | `preferenceRange` | `preference_range` |
| not converged | warns, labels all -1 | warns, keeps the clusters | R's |
| -Inf similarities (never link a pair) | no | yes | yes |
| sparse similarities | no | yes | [roadmap](ROADMAP.md) |
| leveraged AP (large n) | no | `apclusterL` | [roadmap](ROADMAP.md) |
| exemplar-based agglomerative clustering | no | `aggExCluster` | [roadmap](ROADMAP.md) |

scikit-learn's defaults often stop before convergence. In the benchmark below, at 2,000 and 4,000
points they hit the 200-iteration limit, while R's defaults converge.

`AffinityPropagation` passes scikit-learn's `check_estimator`.

## Benchmark

`uv run benchmark.py` runs naffprop and `sklearn.cluster.affinity_propagation` on the same
similarity matrix (blobs, 10 centers) with the same preference and parameters. Each case runs in
its own process, which measures how much memory the fit needs on top of that matrix. One run on
a 4-core Linux container (the timings are noisy from run to run):

| n | parameters | naffprop (s) | scikit-learn (s) | speedup | naffprop memory (MiB) | scikit-learn memory (MiB) | clusters | iterations |
|--:|:-----------|---------:|-------------:|--------:|----------------:|--------------------:|---------:|-----------:|
| 500 | sklearn defaults | 0.08 | 0.15 | 1.8x | 6 | 8 | 13 / 13 | 54 / 54 |
| 500 | R defaults | 0.29 | 0.30 | 1.0x | 6 | 8 | 10 / 10 | 134 / 134 |
| 1,000 | sklearn defaults | 0.67 | 1.73 | 2.6x | 23 | 31 | 20 / 19 | 129 / 184 |
| 1,000 | R defaults | 1.32 | 1.95 | 1.5x | 23 | 32 | 16 / 16 | 206 / 206 |
| 2,000 | sklearn defaults | 4.76 | 7.67 | 1.6x | 93 | 126 | 70 / 118 | 200 / 200 |
| 2,000 | R defaults | 4.03 | 6.04 | 1.5x | 93 | 125 | 27 / 27 | 147 / 147 |
| 4,000 | sklearn defaults | 23.43 | 54.49 | 2.3x | 376 | 603 | 717 / 1863 | 200 / 200 |
| 4,000 | R defaults | 19.38 | 46.49 | 2.4x | 377 | 494 | 37 / 37 | 168 / 168 |

naffprop needs three n x n matrices (a copy of the similarities, the responsibilities and the
availabilities), as R does. scikit-learn needs four or five. When both converge they find the
same clusters. When they don't, each stops at a different point, because each draws its own noise.

## How it is built

```
pyproject.toml
nimlang.lock                    # pinned commits of the Nim dependencies
src/naffprop/matrix.nim         # a dense square matrix: all the linear algebra AP needs
src/naffprop/ap.nim             # the algorithm and the preference range, plain Nim
src/naffprop/core.nim           # nimpy exports, importable as naffprop.core
src/naffprop/__init__.py        # R-style functions: argument checks, preferences, apcluster_k
src/naffprop/_estimator.py      # the scikit-learn estimator
tests/reference.py              # numpy port of Frey and Dueck's MATLAB code
benchmark.py                    # naffprop vs scikit-learn (a uv script: deps in its header)
```

Affinity propagation needs no linear algebra library. Its messages are elementwise updates plus
row maxima and column sums, so `matrix.nim` is a 20-line row-major `seq[float]`. Nothing depends
on Arraymancer or BLAS, which keeps the door open to Nim's JS backend.
[nimpy-numpy](https://github.com/pietroppeter/nimpy-numpy) passes numpy arrays to Nim.

naffprop is not on PyPI yet. Install it from GitHub; uv builds it with nimlang, so no Nim and
no C compiler are needed:

```sh
uv add git+https://github.com/pietroppeter/naffprop
```

```sh
uv sync                  # builds the extension
uv run pytest tests      # against the numpy reference and scikit-learn
uv run benchmark.py      # naffprop vs scikit-learn
```

## References

- Frey and Dueck, [Clustering by Passing Messages Between Data Points](https://doi.org/10.1126/science.1136800), Science 2007
- [apcluster] (R) by Bodenhofer, Kothmeier and Hochreiter, and its [paper](https://doi.org/10.1093/bioinformatics/btr406)
- [scikit-learn's AffinityPropagation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html)

The project started during an [Open Source Saturday](https://www.meetup.com/it-IT/Open-Source-Saturday-Milano/)
in 2023, as an experiment in using Nim as a Cython alternative; that first version is in the
git history but it never went anywhere. It restarted with Claude in 2026.

[Affinity propagation]: https://en.wikipedia.org/wiki/Affinity_propagation
[apcluster]: https://cran.r-project.org/package=apcluster
[Nim]: https://nim-lang.org
