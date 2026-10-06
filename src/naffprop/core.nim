## The Python module naffprop.core: thin nimpy wrappers around ap.nim.
## Arguments are checked by the Python package (naffprop/__init__.py).

import std/math
import nimpy, nimpy_numpy
import ap

proc toMatrix(a: NumpyArray[float64]): Matrix =
  ## A copy of a square 2D array (the algorithm writes the preferences and the
  ## noise into it, and numpy's memory must not change).
  let n = a.shape[0]
  result = zeros(n)
  for i in 0 ..< n:
    for j in 0 ..< n:
      result[i, j] = a[i, j]

proc apcluster(s, p: NumpyArray[float64], maxits, convits: int,
               damping: float, noise: bool, seed: int64):
    (seq[int], seq[int], int, bool, float, float, float) {.exportpy.} =
  ## Affinity propagation on the similarities `s` (n x n) with preferences
  ## `p` (1 or n values). Returns (exemplars, labels, iterations, converged,
  ## netsim, dpsim, expref).
  var m = s.toMatrix
  var pref = newSeq[float](p.len)
  for i in 0 ..< p.len: pref[i] = p[i]
  m.prepare(pref, noise, seed)
  let r = affinityPropagation(m, maxits, convits, damping)
  (r.exemplars, r.labels, r.iterations, r.converged, r.netsim, r.dpsim,
   r.expref)

proc preferenceRange(s: NumpyArray[float64], exact: bool): (float, float) {.exportpy.} =
  ## (pmin, pmax): see ap.preferenceRange.
  ap.preferenceRange(s.toMatrix, exact)

proc negDistMat(x: NumpyArray[float64], r: float): NumpyArray[float64] {.exportpy.} =
  ## -d(i, j)^r, with d the Euclidean distance between the rows of `x`
  ## (n x d).
  let n = x.shape[0]
  let d = x.shape[1]
  var rows = newSeq[float](n * d)
  for i in 0 ..< n:
    for k in 0 ..< d:
      rows[i * d + k] = x[i, k]
  result = newNumpyArray[float64](n, n)
  for i in 0 ..< n:
    result[i, i] = 0
    for j in i + 1 ..< n:
      var t = 0.0
      for k in 0 ..< d:
        let v = rows[i * d + k] - rows[j * d + k]
        t += v * v
      let v = if r == 2: -t elif r == 1: -sqrt(t) else: -pow(t, r / 2)
      result[i, j] = v
      result[j, i] = v
