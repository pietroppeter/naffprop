# The Nim package on its own, with no Python: `import naffprop` as a Nim user
# would. Runs with the C and JS backends (CI's `nim` job):
#   nim c -r --path:src tests/test_nim.nim
#   nim js -d:nodejs -r --path:src tests/test_nim.nim

import std/[math, unittest]
import naffprop

# Three groups of points on a line: AP finds one exemplar in each (the noise
# breaks the ties between equally spaced points).
const xs = [0.0, 0.1, 0.2, 5.0, 5.1, 5.2, 10.0, 10.1, 10.2]

proc negSqDist(): Matrix[float] =
  result = zeros[float](xs.len)
  for i in 0 ..< xs.len:
    for j in 0 ..< xs.len:
      result[i, j] = -((xs[i] - xs[j]) ^ 2)

suite "naffprop":
  test "dense":
    var s = negSqDist()
    let v = s.view
    let (lo, hi) = preferenceRange(v)
    check lo < hi
    v.prepare([-1.0], noise = true, seed = 42)
    let r = v.affinityPropagation(damping = 0.5)
    check r.converged
    check r.exemplars == @[1, 4, 7]
    check r.labels == @[0, 0, 0, 1, 1, 1, 2, 2, 2]

  test "pruned (ScaleAP) finds the same exemplars":
    var s = negSqDist()
    let v = s.view
    v.prepare([-1.0], noise = true, seed = 42)
    check v.affinityPropagationPruned(damping = 0.5).exemplars == @[1, 4, 7]

  test "sparse, every pair stored, finds the same exemplars":
    var s = SparseMatrix[float](n: xs.len)
    for i in 0 ..< xs.len:
      s.rowStart.add s.col.len
      for j in 0 ..< xs.len:
        if i == j: s.diag.add s.col.len
        s.col.add j
        s.val.add (if i == j: -1.0 else: -((xs[i] - xs[j]) ^ 2))
    s.rowStart.add s.col.len
    s.prepare(noise = true, seed = 42)
    check s.affinityPropagation(damping = 0.5).exemplars == @[1, 4, 7]
