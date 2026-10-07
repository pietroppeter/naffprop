## Leveraged affinity propagation, as R's apclusterL: the similarities of all
## n points to a sample of m of them (an n x m matrix), and only the sampled
## points can be exemplars. Memory and time grow with n * m instead of n^2.
##
## It is affinity propagation on the pairs (i, sel[c]) and the diagonal: the
## updates are those of `ap.affinityPropagation` on these pairs, with the
## self-similarities (the preferences) kept apart from the matrix.

import std/[fenv, math]
import ap, rng

type
  RectView*[T: SomeFloat] = object
    ## An n x m matrix, stored by row where numpy stores it.
    n*, m*: int
    data*: ptr UncheckedArray[T]

template `[]`*[T](s: RectView[T]; i, c: int): T = s.data[i * s.m + c]
template `[]=`*[T](s: RectView[T]; i, c: int; v: T) = s.data[i * s.m + c] = v

template maxFloat(T): untyped = maximumPositiveValue(T)

proc prepare*[T](s: RectView[T], noise: bool, seed: int64) =
  ## Noise and -Inf/NaN to -maxFloat, in place, as `ap.prepare`.
  var r = initRng(seed)
  for k in 0 ..< s.n * s.m:
    var x = s.data[k]
    if noise: x += T((epsilon(T) * x + minimumPositiveValue(T) * 100) * r.gauss())
    if x.isNaN or x < -maxFloat(T): x = -maxFloat(T)
    s.data[k] = x

{.push checks: off.}  # hot loops: indices are in range by construction

proc affinityPropagation*[T](s: RectView[T], sel: seq[int], p: seq[float],
                             maxits = 1000, convits = 100,
                             damping = 0.9): ApResult =
  ## Affinity propagation with similarities s[i, c] of point i to point
  ## sel[c] (sel increasing), and preferences p (one per point). Exemplars
  ## are among the points in sel; s[sel[c], c] is not used.
  let n = s.n
  let m = s.m
  let lam = T(damping)
  var
    r = newSeq[T](n * m)       # responsibilities of i to sel[c], by row
    a = newSeq[T](n * m)       # availabilities
    rd = newSeq[T](n)          # self-responsibilities r(i, i)
    ad = newSeq[T](n)          # self-availabilities a(i, i)
    pt = newSeq[T](n)          # preferences
    selCol = newSeq[int](n)    # column of each point in s, or -1
    colsum = newSeq[float](m)
    conv = initConvergence(n, convits)
    isEx = newSeq[bool](n)
    it = 0
  for i in 0 ..< n:
    pt[i] = T(p[i])
    selCol[i] = -1
  for c, j in sel: selCol[j] = c
  while true:
    # Responsibilities, by row, as in ap.nim: the competitors of a column are
    # the other sampled points and i itself (with its preference).
    for c in 0 ..< m: colsum[c] = 0
    for i in 0 ..< n:
      var max1 = ad[i] + pt[i]
      var max2 = T(-Inf)
      var cmax = -1                      # -1: the diagonal
      for c in 0 ..< m:
        if c == selCol[i]: continue
        let v = a[i * m + c] + s[i, c]
        if v > max1:
          max2 = max1
          max1 = v
          cmax = c
        elif v > max2:
          max2 = v
      for c in 0 ..< m:
        if c == selCol[i]: continue
        let v = (1 - lam) * (s[i, c] - (if c == cmax: max2 else: max1)) +
                lam * r[i * m + c]
        r[i * m + c] = min(v, maxFloat(T))
        if r[i * m + c] > 0: colsum[c] += r[i * m + c]
      let v = (1 - lam) * (pt[i] - (if cmax == -1: max2 else: max1)) + lam * rd[i]
      rd[i] = min(v, maxFloat(T))
    # Availabilities. A sampled point's column sum also counts its own
    # responsibility r(j, j); a point not sampled is linked to no one, so its
    # self-availability stays 0.
    for c, j in sel: colsum[c] += rd[j]
    for i in 0 ..< n:
      for c in 0 ..< m:
        if c == selCol[i]: continue
        let v = r[i * m + c]
        var x = colsum[c] - (if v > 0: float(v) else: 0.0)
        if x > 0: x = 0
        a[i * m + c] = (1 - lam) * T(x) + lam * a[i * m + c]
    for c, j in sel:
      ad[j] = (1 - lam) * T(colsum[c] - rd[j]) + lam * ad[j]
    for i in 0 ..< n: isEx[i] = ad[i] + rd[i] > 0
    if conv.update(it, isEx, maxits): break
    inc it
  result = conv.toResult(it)

  # Exemplars: only the sampled ones count, as in R.
  var ex: seq[int]                   # exemplar of each cluster
  for i in 0 ..< n:
    if isEx[i] and selCol[i] >= 0: ex.add i
  if ex.len == 0:
    result = initConvergence(n, convits).toResult(it)
    result.converged = false
    return
  var c = newSeq[int](n)             # cluster of each point
  var exPos = newSeq[int](n)
  template sim(i, j: int): float =
    ## Similarity of i to the sampled point j, the preference if i == j.
    (if i == j: float(pt[i]) else: float(s[i, selCol[j]]))
  proc assign() =
    for j in 0 ..< n: exPos[j] = -1
    for e, j in ex: exPos[j] = e
    for i in 0 ..< n:
      if exPos[i] >= 0:
        c[i] = exPos[i]
        continue
      var best = -Inf
      c[i] = 0
      for e, j in ex:
        if sim(i, j) > best:
          best = sim(i, j)
          c[i] = e
  assign()
  # Refinement, as in ap.nim, among the sampled members of each cluster.
  var members = newSeq[seq[int]](ex.len)
  for i in 0 ..< n: members[c[i]].add i
  for e in 0 ..< ex.len:
    var best = -Inf
    for j in members[e]:
      if selCol[j] < 0: continue
      var t = 0.0
      for i in members[e]: t += sim(i, j)
      if t > best:
        best = t
        ex[e] = j
  assign()

  result.exemplars.setLen 0
  for j in 0 ..< n:
    if exPos[j] >= 0: result.exemplars.add j
  var pos = newSeq[int](n)
  for e, j in result.exemplars: pos[j] = e
  for i in 0 ..< n:
    let j = ex[c[i]]
    result.labels[i] = pos[j]
    if i == j: result.expref += sim(i, i)
    else: result.dpsim += sim(i, j)
  result.netsim = result.dpsim + result.expref

{.pop.}
