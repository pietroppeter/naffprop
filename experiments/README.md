# Experiments

Scripts that reproduce the evidence behind naffprop's design decisions. They are not part of the
package or of CI. Run them from the repository root with `uv run python experiments/<script>`.

Clusterings are compared by **net similarity**, the objective that affinity propagation
maximizes: the sum of each point's similarity to its exemplar plus the preferences of the
exemplars (higher is better). Two runs with the same similarities and preferences can be compared
by it directly. The adjusted Rand index (ARI) is given as well. It measures how much two
clusterings agree, from 1 (the same partition) to 0 (no more than chance).

The results below come from the cloud container described in the README (Intel Xeon, 4 cores).

## ScaleAP's equations: `paper_equations.py`

Decision: `scaleap=True` applies ScaleAP's pruning to Frey and Dueck's updates, not the paper's
own equations.

The paper (Shiokawa, AAAI 2021) states affinity propagation with a self-responsibility that
ignores the availabilities, s(k, k) - max s(k, j), and with availabilities computed from the
previous iteration's responsibilities. In numpy, on blobs, with the same similarities and the
median preference:

| n | centers | damping | clusters (Frey-Dueck / paper) | net similarity (Frey-Dueck / paper) | ARI paper vs Frey-Dueck | pruned: same exemplars | pruned: same iterations | pairs stored (% of n^2) |
|--:|--:|--:|--:|--:|--:|:-:|:-:|--:|
| 150 | 3 | 0.9 | 9 / 10 | -187.4 / -189.1 | 0.69 | yes | yes | 9.9 |
| 150 | 3 | 0.5 | 10 / 143 | -188.0 / -1389.3 | 0.01 | yes | yes | 13.8 |
| 300 | 5 | 0.9 | 10 / 10 | -695.4 / -742.6 | 0.71 | yes | yes | 10.1 |
| 300 | 5 | 0.5 | 10 / 260 | -686.8 / -9358.0 | 0.08 | yes | yes | 18.8 |
| 400 | 8 | 0.9 | 9 / 11 | -1106.3 / -1166.1 | 0.79 | yes | yes | 8.8 |
| 400 | 8 | 0.5 | 12 / 384 | -1151.5 / -18834.0 | 0.01 | yes | yes | 17.8 |

- The paper's equations reach a lower net similarity: 1-7% lower with damping 0.9. With damping
  0.5 they don't converge, and almost every point ends up as its own exemplar.
- The pruning applied to Frey and Dueck's updates ("pruned", what `scaleap.nim` does) finds the
  same exemplars in the same number of iterations. At its peak it stores the messages of 9-19% of
  the pairs at these small n; at n = 4,000 in `benchmark.py` it was about 4%.

## ScaleAP's C++ code: `scaleap_cpp.py`

The authors' reference code, [LazyShion/ScaleAP](https://github.com/LazyShion/ScaleAP) (MIT),
run on the same similarities and preference as naffprop, without noise. The script clones it at
a fixed commit into `experiments/.scaleap` and builds it with `make` (git, make and g++ needed).

| n | damping | clusters naffprop / C++ | net similarity naffprop / C++ | ARI | naffprop (s) | scaleap=True (s) | C++ (s) |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 300 | 0.9 | 12 / 13 | -610.5 / -642.2 | 0.64 | 0.07 | 0.05 | 12.5 |
| 300 | 0.5 | 14 / 297 | -625.1 / -8040.1 | 0.00 | 0.05 | 0.04 | 33.4 |
| 600 | 0.9 | 10 / 13 | -1404.2 / -1470.9 | 0.74 | 0.27 | 0.19 | 153.6 |
| 600 | 0.5 | 16 / 598 | -1496.5 / -31709.8 | 0.00 | 0.36 | 0.26 | 275.8 |

Does the C++ code converge? n = 300, damping 0.5: the points whose exemplar differs from the run
one iteration shorter.

| iterations | clusters | points changed |
|--:|--:|--:|
| 399 | 272 | 29 |
| 400 | 272 | 50 |
| 401 | 237 | 83 |
| 402 | 232 | 114 |

- The C++ code behaves like the paper's equations. With damping 0.9 its net similarity is 5%
  lower. With damping 0.5, its default, it does not converge: between consecutive iterations,
  30 to 114 of the 300 points change exemplar after 400 iterations.
- It is slower than the dense loop by two to three orders of magnitude. Its availability update
  loops over all points for every candidate pair, so its time grew about 13x when n doubled.

## The options for large n: `scaling.py`

Decision: document `scaleap`, sparse and leveraged as the options for large n, with their
trade-offs (README, "Scaling to large n").

Blobs with 10 centers and the same preference as `benchmark.py`:

| n | option | clusters | net similarity | below the full run | ARI vs full |
|--:|:--|--:|--:|--:|--:|
| 4,000 | full | 37 | -5245.3 | 0.0% | 1.00 |
| 4,000 | scaleap | 37 | -5245.3 | 0.0% | 1.00 |
| 4,000 | sparse, 10% neighbours | 37 | -5245.3 | 0.0% | 1.00 |
| 4,000 | leveraged, 10% sample | 36 | -5378.1 | 2.5% | 0.69 |
| 8,000 | full | 55 | -7450.5 | 0.0% | 1.00 |
| 8,000 | scaleap | 55 | -7450.5 | 0.0% | 1.00 |
| 8,000 | sparse, 10% neighbours | 55 | -7450.5 | 0.0% | 1.00 |
| 8,000 | leveraged, 10% sample | 57 | -7560.8 | 1.5% | 0.60 |

Leveraged AP agrees less with the full clustering (ARI 0.6-0.7) than its net similarity
suggests: it is only 1.5-2.5% lower. With the median preference, 37 to 55 clusters cut the 10
blobs into pieces, and where the cuts fall changes little in net similarity.
