# The Nim loop next to the original

Frey and Dueck's MATLAB code (apcluster.m) computes each message with whole-matrix operations.
[`tests/reference.py`](../tests/reference.py) is a line-by-line numpy port of it, and the tests
check that naffprop finds the same exemplars, labels and number of iterations.
`affinityPropagation` in [`src/naffprop/ap.nim`](../src/naffprop/ap.nim) computes the same
messages with loops, one row at a time, so that it needs no n x n temporaries.

## Step by step

| step | original (numpy port of the MATLAB) | naffprop (Nim) |
|:--|:--|:--|
| largest and second largest of a(i, k) + s(i, k) in each row | `as_ = a + s` (an n x n temporary), `argmax`, set the max to -Inf, `max` again | one pass over the row keeps `max1`, `max2` and `jmax` |
| new responsibilities | `rnew = s - m1[:, None]`, then the `jmax` column of each row uses `m2` (one n x n temporary) | same formula, element by element |
| damping | `r = (1 - lam) * rnew + lam * r` (one more temporary) | in the same loop, into `r` |
| column sums of the positive responsibilities | `rp = max(r, 0)` with r's diagonal (a temporary), `rp.sum(axis=0)` | added to `colsum` while the responsibilities are written |
| new availabilities | `anew = colsum - rp`, `min(anew, 0)` off the diagonal (one more temporary) | same formula, element by element |
| damping | `a = (1 - lam) * anew + lam * a` | in the same loop, into `a` |
| convergence | an n x convits boolean matrix `e`, summed by row at every iteration | the same history as a ring buffer, plus a running count per point, so no sum over convits |
| refinement and labels | unchanged | unchanged |

## What it costs to read, and what it buys

The Nim loop is about 45 lines; the original's iteration is about 15. The extra lines come from
three changes, each worth its memory or time:

- **No temporaries.** The original allocates four or five n x n matrices per iteration
  (`as_`, `rnew`, `rp`, `anew`, plus the damping results). The loop keeps only the similarities,
  the responsibilities and the availabilities: 3 n x n matrices, 2 when the similarities are not
  copied (`copy=False`). This is what lets naffprop need a third to three quarters of
  scikit-learn's memory.
- **Fused passes.** The column sums are taken while writing the responsibilities, so each
  iteration reads the matrices twice, not five times. The loop is memory-bound, so this is most of
  the speed.
- **Running convergence count.** O(n) per iteration instead of O(n x convits).

None of them changes a result: the formulas are the same, element by element, and the tests
compare with the reference at every damping, preference and dataset they use.

## Keep both?

The readable version already exists, as `tests/reference.py`, and it is executable and tested
against the Nim code. A second Nim implementation written like the MATLAB would need a
whole-matrix layer (`+`, `max`, `sum` by axis) that the fast code does not use. It would also be
one more implementation to keep identical.

The planned explained implementation (see [ROADMAP.md](../ROADMAP.md)) is a better place for
the readable version: a nimib document can show each MATLAB line next to the Nim lines that
replace it, as in the table above, and run both on a small example.
