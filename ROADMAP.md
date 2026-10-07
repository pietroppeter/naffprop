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

## Next: from R's apcluster

- **Sparse similarities**: a similarity matrix with only the pairs worth linking (e.g. the k
  nearest neighbours), so memory grows with the number of pairs, not n^2. Needs a sparse
  (COO/CSR) matrix type in naffprop and messages on the stored pairs only. In the estimator:
  accept a `scipy.sparse` matrix with `affinity="precomputed"`.
- **Leveraged affinity propagation** (R's `apclusterL`): runs on a random fraction of the
  columns of the similarity matrix, several sweeps, keeping the best net similarity. Works from
  a similarity function, so the full matrix is never built.
- **Exemplar-based agglomerative clustering** (R's `aggExCluster`): merges the clusters found
  by AP into a hierarchy (dendrogram, `cutree`).
- `details`: the net similarity at every iteration, to monitor convergence (R's `plot`).

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
  AAAI 2021: from O(n^2 T) to O(n T) time (T iterations). The similarities still take O(n^2)
  memory, so it pairs well with sparse similarities.

## Performance

- The dense loop is memory-bound: three n x n float64 matrices are read on every iteration.
  Options: float32 messages, fusing the responsibility and availability passes further, threads
  over rows, SIMD.
- Contiguous fast path for the input copy and `neg_dist_mat` output with nimpy_numpy's
  `toOpenArray`/`unsafeData`, once a nimpy_numpy release includes them.

## Benchmarks

- Add R's apcluster to `benchmark.py` (through `Rscript`, when it is installed).
- Benchmark each new feature: sparse vs dense, leveraged vs full, the scalable variant.

## Packaging

- The JS backend (interactive visualizations): `ap.nim` and `matrix.nim` are plain Nim.
