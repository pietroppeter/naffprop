## Affinity propagation on sparse similarities, as R's apcluster on a sparse
## matrix: messages are passed only between the pairs whose similarity is
## stored, so memory grows with the number of pairs, not n^2. The pairs that
## are not stored have similarity -Inf: they are never linked.
##
## The updates are those of `ap.affinityPropagation`, restricted to the stored
## pairs, so on the same similarities (with -Inf for the pairs not stored) both
## find the same exemplars.

import std/[fenv, math]
import ap, rng

type
  SparseMatrix*[T: SomeFloat] = object
    ## Stored by row (CSR): the pairs of row i are at positions
    ## rowStart[i] ..< rowStart[i + 1], with columns col[k] and similarities
    ## val[k]. Every row stores its diagonal (the preference), at diag[i].
    n*: int
    rowStart*, col*, diag*: seq[int]
    val*: seq[T]

template maxFloat(T): untyped = maximumPositiveValue(T)

proc prepare*[T](s: var SparseMatrix[T], noise: bool, seed: int64) =
  ## Noise and -Inf/NaN to -maxFloat, as `ap.prepare` (the preferences are
  ## already on the diagonal).
  if noise:
    var r = initRng(seed)
    for x in s.val.mitems:
      x += T((epsilon(T) * x + minimumPositiveValue(T) * 100) * r.gauss())
  for x in s.val.mitems:
    if x.isNaN or x < -maxFloat(T): x = -maxFloat(T)

{.push checks: off.}  # hot loops: indices are in range by construction

proc affinityPropagation*[T](s: SparseMatrix[T], maxits = 1000, convits = 100,
                             damping = 0.9): ApResult =
  ## Affinity propagation on the stored pairs of `s`.
  let n = s.n
  let lam = T(damping)
  var
    r = newSeq[T](s.val.len)              # responsibilities, one per stored pair
    a = newSeq[T](s.val.len)              # availabilities
    colsum = newSeq[float](n)
    conv = initConvergence(n, convits)
    isEx = newSeq[bool](n)
    it = 0
  while true:
    # Responsibilities, by row, and the column sums of the positive ones (and
    # the diagonal), as in ap.nim: the max is over the stored pairs only.
    for j in 0 ..< n: colsum[j] = 0
    for i in 0 ..< n:
      var max1, max2 = T(-Inf)
      var kmax = -1
      for k in s.rowStart[i] ..< s.rowStart[i + 1]:
        let v = a[k] + s.val[k]
        if v > max1:
          max2 = max1
          max1 = v
          kmax = k
        elif v > max2:
          max2 = v
      for k in s.rowStart[i] ..< s.rowStart[i + 1]:
        let v = (1 - lam) * (s.val[k] - (if k == kmax: max2 else: max1)) +
                lam * r[k]
        r[k] = min(v, maxFloat(T))
        if r[k] > 0 or k == s.diag[i]: colsum[s.col[k]] += r[k]
    # Availabilities: the same formula as ap.nim, for the stored pairs.
    for i in 0 ..< n:
      for k in s.rowStart[i] ..< s.rowStart[i + 1]:
        let v = r[k]
        let onDiag = k == s.diag[i]
        var x = colsum[s.col[k]] - (if v > 0 or onDiag: float(v) else: 0.0)
        if x > 0 and not onDiag: x = 0
        a[k] = (1 - lam) * T(x) + lam * a[k]
    for i in 0 ..< n: isEx[i] = a[s.diag[i]] + r[s.diag[i]] > 0
    if conv.update(it, isEx, maxits): break
    inc it
  result = conv.toResult(it)
  if conv.k == 0: return

  # The similarity of i to j: stored, or -maxFloat (-Inf) if not.
  var sim = newSeq[float](n)          # of each point to its exemplar
  var c = newSeq[int](n)              # cluster of each point
  var ex: seq[int]                    # exemplar of each cluster
  for i in 0 ..< n:
    if isEx[i]: ex.add i
  var exPos = newSeq[int](n)          # cluster of each exemplar, or -1
  proc assign() =
    ## Each point to its most similar exemplar among those it is linked to
    ## (the first if none), each exemplar to itself.
    for j in 0 ..< n: exPos[j] = -1
    for e, j in ex: exPos[j] = e
    for i in 0 ..< n:
      c[i] = 0
      sim[i] = -maxFloat(T)
      if exPos[i] >= 0:
        c[i] = exPos[i]
        sim[i] = float(s.val[s.diag[i]])
        continue
      for k in s.rowStart[i] ..< s.rowStart[i + 1]:
        let j = s.col[k]
        if exPos[j] >= 0 and float(s.val[k]) > sim[i] and k != s.diag[i]:
          sim[i] = float(s.val[k])
          c[i] = exPos[j]
  assign()
  # Refinement, as in ap.nim: in each cluster, the exemplar becomes the member
  # with the largest sum of similarities from the members, among those linked
  # to every member (if none is, the exemplar stays).
  var members = newSeq[seq[int]](ex.len)
  for i in 0 ..< n: members[c[i]].add i
  var t = newSeq[float](n)
  var linked = newSeq[int](n)
  for e in 0 ..< ex.len:
    for j in members[e]:
      t[j] = 0
      linked[j] = 0
    for i in members[e]:
      for k in s.rowStart[i] ..< s.rowStart[i + 1]:
        let j = s.col[k]
        if c[j] == e:
          t[j] += float(s.val[k])
          inc linked[j]
    var best = -Inf
    for j in members[e]:
      if linked[j] == members[e].len and t[j] > best:
        best = t[j]
        ex[e] = j
  assign()

  result.exemplars.setLen 0
  result.dpsim = 0
  result.expref = 0
  for j in 0 ..< n:
    if exPos[j] >= 0: result.exemplars.add j
  var pos = newSeq[int](n)
  for e, j in result.exemplars: pos[j] = e
  for i in 0 ..< n:
    let j = ex[c[i]]
    result.labels[i] = pos[j]
    if i == j: result.expref += sim[i]
    else: result.dpsim += sim[i]
  result.netsim = result.dpsim + result.expref

{.pop.}
