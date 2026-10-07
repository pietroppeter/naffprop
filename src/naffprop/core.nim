## The Python module naffprop.core: thin nimpy wrappers around ap.nim.
## Arguments are checked by the Python package (naffprop/__init__.py), which
## passes C-contiguous float32 or float64 arrays.

import std/math
import nimpy, nimpy_numpy
import ap, sparse, leveraged

proc view[T](a: NumpyArray[T]): MatrixView[T] =
  ## The elements of a square C-contiguous array, where numpy stores them.
  MatrixView[T](n: a.shape[0], data: a.unsafeData)

proc isFloat32(o: PyObject): bool =
  o.dtype.name.to(string) == "float32"

proc run[T](s: NumpyArray[T], p: seq[float], maxits, convits: int,
            damping: float, noise: bool, seed: int64):
    (seq[int], seq[int], int, bool, float, float, float) =
  let m = s.view
  m.prepare(p, noise, seed)
  let r = affinityPropagation(m, maxits, convits, damping)
  (r.exemplars, r.labels, r.iterations, r.converged, r.netsim, r.dpsim,
   r.expref)

proc apcluster(s: PyObject, p: seq[float], maxits, convits: int,
               damping: float, noise: bool, seed: int64):
    (seq[int], seq[int], int, bool, float, float, float) {.exportpy.} =
  ## Affinity propagation on the similarities `s` (n x n, float32 or float64,
  ## C-contiguous, overwritten: noise and preferences) with preferences `p`
  ## (1 or n values). Returns (exemplars, labels, iterations, converged,
  ## netsim, dpsim, expref).
  if s.isFloat32:
    run(asNumpyArray[float32](s, writable = true), p, maxits, convits, damping, noise, seed)
  else:
    run(asNumpyArray[float64](s, writable = true), p, maxits, convits, damping, noise, seed)

proc toSeqInt(a: NumpyArray[int64]): seq[int] =
  result = newSeq[int](a.size)
  for k, v in a.toOpenArray: result[k] = int(v)

proc runSparse[T](n: int, rowStart, col, diag: seq[int], val: NumpyArray[T],
                  maxits, convits: int, damping: float, noise: bool, seed: int64):
    (seq[int], seq[int], int, bool, float, float, float) =
  var s = SparseMatrix[T](n: n, rowStart: rowStart, col: col, diag: diag,
                          val: @(val.toOpenArray))
  s.prepare(noise, seed)
  let r = affinityPropagation(s, maxits, convits, damping)
  (r.exemplars, r.labels, r.iterations, r.converged, r.netsim, r.dpsim,
   r.expref)

proc apclusterSparse(n: int, rowStart, col, diag: NumpyArray[int64], val: PyObject,
                     maxits, convits: int, damping: float, noise: bool, seed: int64):
    (seq[int], seq[int], int, bool, float, float, float) {.exportpy.} =
  ## Affinity propagation on sparse similarities, by row (CSR): row i holds
  ## the pairs rowStart[i] ..< rowStart[i + 1], with columns `col` and
  ## similarities `val` (float32 or float64), its diagonal at diag[i].
  let (rs, c, d) = (rowStart.toSeqInt, col.toSeqInt, diag.toSeqInt)
  if val.isFloat32:
    runSparse(n, rs, c, d, asNumpyArray[float32](val), maxits, convits, damping, noise, seed)
  else:
    runSparse(n, rs, c, d, asNumpyArray[float64](val), maxits, convits, damping, noise, seed)

proc runLeveraged[T](s: NumpyArray[T], sel: seq[int], p: seq[float], maxits, convits: int,
                     damping: float, noise: bool, seed: int64):
    (seq[int], seq[int], int, bool, float, float, float) =
  let v = RectView[T](n: s.shape[0], m: s.shape[1], data: s.unsafeData)
  v.prepare(noise, seed)
  let r = affinityPropagation(v, sel, p, maxits, convits, damping)
  (r.exemplars, r.labels, r.iterations, r.converged, r.netsim, r.dpsim,
   r.expref)

proc apclusterLeveraged(s: PyObject, sel: seq[int], p: seq[float], maxits, convits: int,
                        damping: float, noise: bool, seed: int64):
    (seq[int], seq[int], int, bool, float, float, float) {.exportpy.} =
  ## Leveraged affinity propagation on the similarities `s` (n x m, float32 or
  ## float64, C-contiguous, overwritten with the noise) of every point to the
  ## points `sel`, with preferences `p` (n values).
  if s.isFloat32:
    runLeveraged(asNumpyArray[float32](s, writable = true), sel, p, maxits, convits, damping, noise, seed)
  else:
    runLeveraged(asNumpyArray[float64](s, writable = true), sel, p, maxits, convits, damping, noise, seed)

proc preferenceRange(s: PyObject, exact: bool): (float, float) {.exportpy.} =
  ## (pmin, pmax): see ap.preferenceRange.
  if s.isFloat32: asNumpyArray[float32](s).view.preferenceRange(exact)
  else: asNumpyArray[float64](s).view.preferenceRange(exact)

proc fillNegDist[T](x: openArray[float64], d: int, r: float, sel: seq[int],
                   dst: var openArray[T]) =
  ## -d(i, sel[c])^r, with d the Euclidean distance between the rows of `x`,
  ## into the n x sel.len matrix `dst`.
  let n = x.len div d
  let m = sel.len
  template negDist(i, j: int): T =
    var t = 0.0
    for k in 0 ..< d:
      let v = x[i * d + k] - x[j * d + k]
      t += v * v
    T(if r == 2: -t elif r == 1: -sqrt(t) else: -pow(t, r / 2))
  if m == n and (n == 0 or sel[^1] == n - 1):  # all the points: symmetric
    for i in 0 ..< n:
      dst[i * n + i] = 0
      for j in i + 1 ..< n:
        let v = negDist(i, j)
        dst[i * n + j] = v
        dst[j * n + i] = v
  else:
    for i in 0 ..< n:
      for c, j in sel: dst[i * m + c] = negDist(i, j)

proc negDistMat(x: NumpyArray[float64], r: float, sel: seq[int], f32: bool): PyObject {.exportpy.} =
  ## -d(i, j)^r between the rows of `x` (n x d, C-contiguous) and the rows
  ## `sel`, as an n x sel.len float32 or float64 array.
  let n = x.shape[0]
  if f32:
    var res = newNumpyArray[float32](n, sel.len)
    fillNegDist(x.toOpenArray, x.shape[1], r, sel, res.toOpenArray)
    res.toPyObject
  else:
    var res = newNumpyArray[float64](n, sel.len)
    fillNegDist(x.toOpenArray, x.shape[1], r, sel, res.toOpenArray)
    res.toPyObject
