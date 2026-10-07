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
AffinityPropagation(affinity="precomputed").fit(S)   # your own similarity matrix, dense or sparse
AffinityPropagation(leveraged=0.1).fit(X)      # large n: R's apclusterL on 10% of the points
AffinityPropagation().fit(X.astype("float32")) # half the memory
```

The same features are also available as functions named after R's:

```python
from naffprop import apcluster, apcluster_k, apcluster_l, neg_dist_mat, preference_range

S = neg_dist_mat(X, r=2)        # -||x_i - x_j||^2, as R's negDistMat
res = apcluster(S, q=0.5)       # APResult: exemplars, labels, clusters, netsim, iterations, converged
res = apcluster_k(S, 5)
pmin, pmax = preference_range(S)
res = apcluster(S_sparse)       # a scipy.sparse matrix: only the stored pairs are linked
res = apcluster_l(X, frac=0.1, sweeps=5)   # leveraged, as R's apclusterL
res = apcluster(S, dtype="float32", copy=False)   # least memory: float32, no copy of S
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
| sparse similarities | no | yes | scipy.sparse, with `affinity="precomputed"` or `apcluster` |
| leveraged AP (large n) | no | `apclusterL` | `leveraged`, `apcluster_l` |
| float32 (half the memory) | no | no | `dtype`, float32 input |
| work in the similarity matrix, no copy | `copy=False` | no | `copy=False` |
| exemplar-based agglomerative clustering | no | `aggExCluster` | [roadmap](ROADMAP.md) |

scikit-learn's defaults often stop before convergence. In the benchmark below, at 2,000 and 4,000
points they hit the 200-iteration limit, while R's defaults converge.

`AffinityPropagation` passes scikit-learn's `check_estimator`.

## Benchmark

`uv run benchmark.py` runs naffprop and `sklearn.cluster.affinity_propagation` on the same
similarity matrix (blobs, 10 centers) with the same preference and parameters. Each case runs
`benchmark_case.py` in its own process, which measures how much memory the fit needs on top of
that matrix (Unix only). It first prints the machine it runs on: paste that with the table when
you share a run. Timings depend on the machine and are noisy from run to run; naffprop has been
1.0-2.6x faster than scikit-learn so far, and needed a third to three quarters of its extra memory.

On an Apple M3 Pro laptop:

- CPU: Apple M3 Pro, 11 cores, 18 GiB RAM
- OS: macOS 15.7.3 (arm64)
- Python 3.13.0, naffprop 0.1.0, numpy 2.5.3, scikit-learn 1.9.1

| n | parameters | naffprop (s) | scikit-learn (s) | speedup | naffprop memory (MiB) | scikit-learn memory (MiB) | clusters | iterations |
|--:|:-----------|---------:|-------------:|--------:|----------------:|--------------------:|---------:|-----------:|
| 500 | sklearn defaults | 0.04 | 0.05 | 1.1x | 5 | 12 | 13 / 13 | 54 / 54 |
| 500 | R defaults | 0.10 | 0.12 | 1.3x | 5 | 12 | 10 / 10 | 134 / 134 |
| 1,000 | sklearn defaults | 0.43 | 0.56 | 1.3x | 25 | 46 | 19 / 21 | 153 / 168 |
| 1,000 | R defaults | 0.58 | 0.67 | 1.2x | 25 | 46 | 16 / 16 | 206 / 206 |
| 2,000 | sklearn defaults | 2.19 | 2.69 | 1.2x | 102 | 250 | 45 / 105 | 200 / 200 |
| 2,000 | R defaults | 1.63 | 2.02 | 1.2x | 102 | 251 | 27 / 27 | 147 / 147 |
| 4,000 | sklearn defaults | 8.44 | 11.30 | 1.3x | 411 | 1006 | 1387 / 1404 | 200 / 200 |
| 4,000 | R defaults | 7.28 | 9.67 | 1.3x | 411 | 1009 | 37 / 37 | 168 / 168 |

On a cloud container:

- CPU: Intel(R) Xeon(R) Processor @ 2.80GHz, 4 cores, 16 GiB RAM
- OS: Linux (x86_64)
- Python 3.13, naffprop 0.1.0, numpy 2.5.3, scikit-learn 1.9.1

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

With `dtype=np.float32` (or float32 input) the three matrices take half the memory, and with
`copy=False` the similarities are not copied, so two are left: in the benchmark below, about a
quarter of the extra memory scikit-learn needs. On the test data float32 finds the same exemplars as float64. float16 is
not offered: run through the numpy port of the MATLAB code, it changed the clusters on 2 of 6
data sets, as soon as the similarities left its range of about 6.5e4 or n reached 1,000. The
[readability notes](docs/readability.md) compare the Nim loop with the original, step by step.

## How it is built

```
pyproject.toml
nimlang.lock                    # pinned commits of the Nim dependencies
src/naffprop/matrix.nim         # a dense square matrix: all the linear algebra AP needs
src/naffprop/ap.nim             # the algorithm and the preference range, plain Nim
src/naffprop/sparse.nim         # the algorithm on the stored pairs of a sparse matrix
src/naffprop/leveraged.nim      # leveraged AP: the similarities to a sample of the points
src/naffprop/rng.nim            # the seeded noise generator (or std/random)
src/naffprop/core.nim           # nimpy exports, importable as naffprop.core
src/naffprop/__init__.py        # R-style functions: argument checks, preferences, apcluster_k
src/naffprop/_estimator.py      # the scikit-learn estimator
tests/reference.py              # numpy port of Frey and Dueck's MATLAB code
benchmark.py                    # naffprop vs scikit-learn (a uv script: deps in its header)
benchmark_case.py               # one case in its own process: time and peak memory
```

Affinity propagation needs no linear algebra library. Its messages are elementwise updates plus
row maxima and column sums, so `matrix.nim` is a 20-line row-major `seq[float]`. Nothing depends
on Arraymancer or BLAS, which keeps the door open to Nim's JS backend.
[nimpy-numpy](https://github.com/pietroppeter/nimpy-numpy) passes numpy arrays to Nim.

The tiny noise that breaks ties comes from a 20-line seeded generator in `rng.nim` (splitmix64
and Box-Muller), not from `std/random`, which does not cross-compile for macOS with nimlang yet
(it links macOS's Security framework). It also makes a seed give the same clusters on every
platform. To build with `std/random` instead, set `NAFFPROP_STD_RANDOM=1` when building (it
compiles with `-d:naffpropStdRandom`).

Install it from PyPI:

```sh
uv add naffprop        # or: pip install naffprop
```

Wheels are built for Linux (x86_64, aarch64), macOS (arm64, x86_64) and Windows (x86_64), all
cross-compiled by nimlang from one Linux CI job; one wheel per platform serves every Python
from 3.9. Elsewhere pip and uv build the sdist, which needs nimlang but no Nim and no C
compiler. To work on naffprop itself:

```sh
uv sync                  # builds the extension
uv run pytest tests      # against the numpy reference and scikit-learn
uv run benchmark.py      # naffprop vs scikit-learn
```

## References

- Frey and Dueck, [Clustering by Passing Messages Between Data Points](https://doi.org/10.1126/science.1136800), Science 2007
- [apcluster] (R) by Bodenhofer, Kothmeier and Hochreiter, and its [paper](https://doi.org/10.1093/bioinformatics/btr406)
- [scikit-learn's AffinityPropagation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html)

The source code naffprop was compared with, to check what changed since:

| | version compared | source |
|:--|:--|:--|
| R apcluster | 1.4.14 (2025-09-09) | [GitHub mirror of CRAN](https://github.com/cran/apcluster): [`apcluster`](https://github.com/cran/apcluster/blob/master/R/apcluster-methods.R) and its [C++ loop](https://github.com/cran/apcluster/blob/master/src/apclusterC.cpp), [`apclusterK`](https://github.com/cran/apcluster/blob/master/R/apclusterK-methods.R), [`preferenceRange`](https://github.com/cran/apcluster/blob/master/R/preferenceRange-methods.R) and its [C++](https://github.com/cran/apcluster/blob/master/src/preferenceRangeC.cpp), [NEWS](https://github.com/cran/apcluster/blob/master/inst/NEWS) |
| scikit-learn | 1.9.1 | [`sklearn/cluster/_affinity_propagation.py`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/cluster/_affinity_propagation.py) |

The project started during an [Open Source Saturday](https://www.meetup.com/it-IT/Open-Source-Saturday-Milano/)
in 2023, as an experiment in using Nim as a Cython alternative; that first version is in the
git history but it never went anywhere. It restarted with Claude in 2026.

[Affinity propagation]: https://en.wikipedia.org/wiki/Affinity_propagation
[apcluster]: https://cran.r-project.org/package=apcluster
[Nim]: https://nim-lang.org
