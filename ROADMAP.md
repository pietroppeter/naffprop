# Roadmap

Goal: a scikit-learn compatible affinity propagation that takes the best of R's apcluster, then
a scalable version from the literature, with everything benchmarked.

## Done

- Dense affinity propagation in Nim, with R's defaults, `q`, per-point preferences and -Inf
  similarities, matching a numpy port of Frey and Dueck's MATLAB code and scikit-learn.
- `preference_range` and `apcluster_k` / `n_clusters` (R's preferenceRange and apclusterK).
- `AffinityPropagation`, passing scikit-learn's `check_estimator`.
- `benchmark.py` against scikit-learn.
- Released on PyPI, with wheels for 5 platforms cross-built by nimlang and tested on each OS.
- Sparse similarities (R's apcluster on a sparse matrix): scipy.sparse input, messages on the
  stored pairs only.
- Leveraged affinity propagation (R's `apclusterL`): `apcluster_l` and `leveraged`.
- float32 (`dtype`, float32 input) and `copy=False`: from 3 n x n float64 matrices down to 2
  float32 ones. float16 was tried on the numpy port of the MATLAB code and dropped: it changed
  the clusters when the similarities exceeded its range or n reached 1,000.
- The contiguous fast path of nimpy_numpy 0.2.0: similarities used where numpy stores them.
- ScaleAP's pruning (`scaleap=True`): the same exemplars as the dense loop, storing the messages
  of a few percent of the pairs only, about 3x faster.

## Next: from R's apcluster

- **Exemplar-based agglomerative clustering** (R's `aggExCluster`): merges the clusters found
  by AP into a hierarchy (dendrogram, `cutree`).
- `details`: the net similarity at every iteration, to monitor convergence (R's `plot`).
- `apcluster_k` and `preference_range` on sparse similarities.

## Explained implementation

An AI-driven implementation in a scientific context should also be explained and motivated, not
only correct: documents that walk through what the code does and why each choice was made, so
that a reader can understand it, check it and trust it. For naffprop, ideally interactive
documents written in Nim with [nimib](https://github.com/pietroppeter/nimib), for example:

- the message updates on a small example, with responsibilities and availabilities shown at
  each iteration and the exemplars emerging;
- why each choice: damping and R's defaults, the noise that breaks ties, the convergence test,
  the refinement of the exemplars, the preference and its range, what differs from scikit-learn
  and R and why;
- the same for each new feature (sparse, leveraged, the scalable variant).

Context:
- [If math is more than proof, we need to better celebrate the rest of it](https://terrytao.wordpress.com/2026/09/18/if-math-is-more-than-proof-we-need-to-better-celebrate-the-rest-of-it/)
  (Grant Sanderson's guest post on Terence Tao's blog), on giving motivated explanations the
  standing of proofs.
- Simon Willison, [My answers to the questions I posed about porting open source code with LLMs](https://simonwillison.net/2026/Jan/11/answers/),
  including where the value of a library mostly written by AI lies: in how much it is used.

## Then: scalable affinity propagation

- Shiokawa, [*Scalable affinity propagation for massive datasets*](https://ojs.aaai.org/index.php/AAAI/article/view/17160),
  AAAI 2021. Reference implementation in C++ (MIT): [LazyShion/ScaleAP](https://github.com/LazyShion/ScaleAP).
  Its pruning is in (`scaleap=True`, see the README), applied to Frey and Dueck's updates so that
  the exemplars stay the same. What is left:
  - Each iteration still reads every row of similarities, to find each point's best candidate.
    The availabilities off the diagonal are never positive, so with each row sorted once (an
    index as large as float32 similarities) that search could stop at the first similarity below
    the second best value found, and an iteration would read a few similarities per point.
  - ScaleAP on sparse similarities: together, neither the similarities nor the messages would
    take O(n^2) memory.
  - Whether `scaleap=True` should become the default, once it has been run on more data.

## Performance

- The dense loop is memory-bound: three n x n matrices are read on every iteration. Options left:
  threads over rows, SIMD. float32 halves the memory but is not faster yet (the column sums are
  accumulated in float64).
- Compare the speed of float32 and float64 on Apple silicon, where memory bandwidth differs.

## Benchmarks

- Add R's apcluster to `benchmark.py` (through `Rscript`, when it is installed).

## Packaging

- Make `std/random` the default in `rng.nim` (today behind `-d:naffpropStdRandom`) and drop
  the seeded generator (splitmix64 and Box-Muller, used only for the tie-breaking noise), once
  nimlang cross-compiles `std/random` for macOS (it imports `std/sysrand`, which links macOS's
  Security framework).
- The JS backend (interactive visualizations): `ap.nim` and `matrix.nim` are plain Nim.
