## Affinity propagation (Frey and Dueck, Science 2007) on a dense similarity
## matrix.
##
## The algorithm is the one of Frey and Dueck's reference MATLAB code, which
## both R's apcluster and scikit-learn follow: the same message updates,
## damping, convergence test and final refinement of the exemplars. The
## preference range is the one of Frey and Dueck's preferenceRange.m.
##
## Pure Nim (no nimpy): `core.nim` exposes it to Python.

import std/[fenv, math, random]
import matrix

export matrix

const maxFloat = maximumPositiveValue(float)

type
  ApResult* = object
    exemplars*: seq[int]  ## indices of the exemplars, increasing
    labels*: seq[int]     ## for each point, the position of its exemplar in
                          ## `exemplars`; all -1 if there are no exemplars
    iterations*: int
    converged*: bool
    dpsim*: float         ## sum of the similarities of points to their exemplar
    expref*: float        ## sum of the preferences of the exemplars
    netsim*: float        ## dpsim + expref, the objective AP maximizes

proc addNoise*(s: var Matrix, seed: int64) =
  ## Adds a tiny gaussian noise to every similarity, as R and scikit-learn do,
  ## so that ties (e.g. duplicated points) don't make the messages oscillate.
  var r = initRand(seed)
  for x in s.data.mitems:
    x += (epsilon(float) * x + minimumPositiveValue(float) * 100) * r.gauss()

proc setPreferences*(s: var Matrix, p: openArray[float]) =
  ## Puts the preferences on the diagonal: one for every point, or the same
  ## for all if `p` has a single value.
  for i in 0 ..< s.n:
    s[i, i] = if p.len == 1: p[0] else: p[i]

proc prepare*(s: var Matrix, p: openArray[float], noise: bool, seed: int64) =
  ## Readies a copy of the user's similarities for `affinityPropagation`:
  ## noise, preferences on the diagonal, -Inf and NaN to -maxFloat (as R does).
  if noise: s.addNoise(seed)
  s.setPreferences(p)
  for x in s.data.mitems:
    if x.isNaN or x < -maxFloat: x = -maxFloat

{.push checks: off.}  # hot loops: indices are in range by construction

proc affinityPropagation*(s: Matrix, maxits = 1000, convits = 100,
                          damping = 0.9): ApResult =
  ## Affinity propagation on the similarities `s`, whose diagonal holds the
  ## preferences. Stops when the exemplars have not changed for `convits`
  ## iterations, or after `maxits` iterations. Similarities must be finite:
  ## `prepare` turns -Inf (a pair never to link) into -maxFloat.
  let n = s.n
  let lam = damping
  var
    r = zeros(n)                           # responsibilities
    a = zeros(n)                           # availabilities
    colsum = newSeq[float](n)
    hist = newSeq[uint8](n * convits)      # was i an exemplar, last convits its
    count = newSeq[int](n)                 # ... and how many times
    it = 0
    k = 0
    unconverged = false
  while true:
    # Responsibilities, by row: r(i, j) = s(i, j) - max over j' != j of
    # (a(i, j') + s(i, j')). On the way, the column sums of the positive
    # responsibilities (and the diagonal) that the availabilities need.
    for j in 0 ..< n: colsum[j] = 0
    for i in 0 ..< n:
      var max1, max2 = -Inf
      var jmax = 0
      for j in 0 ..< n:
        let v = a[i, j] + s[i, j]
        if v > max1:
          max2 = max1
          max1 = v
          jmax = j
        elif v > max2:
          max2 = v
      for j in 0 ..< n:
        let v = (1 - lam) * (s[i, j] - (if j == jmax: max2 else: max1)) +
                lam * r[i, j]
        r[i, j] = min(v, maxFloat)
        if r[i, j] > 0 or i == j: colsum[j] += r[i, j]
    # Availabilities: a(i, j) = min(0, r(j, j) + sum of the positive r(i', j)
    # for i' not in {i, j}), and a(j, j) = sum of the positive r(i', j), i' != j.
    for i in 0 ..< n:
      for j in 0 ..< n:
        let v = r[i, j]
        var x = colsum[j] - (if v > 0 or i == j: v else: 0.0)
        if x > 0 and i != j: x = 0
        a[i, j] = (1 - lam) * x + lam * a[i, j]
    # Exemplars: a(i, i) + r(i, i) > 0. Converged when no point changed its
    # status in the last convits iterations.
    unconverged = false
    k = 0
    let slot = it mod convits
    for i in 0 ..< n:
      let ex = uint8(a[i, i] + r[i, i] > 0)
      count[i] += ex.int - hist[i * convits + slot].int
      hist[i * convits + slot] = ex
      if count[i] > 0 and count[i] < convits: unconverged = true
      k += ex.int
    if it >= convits - 1 or it >= maxits - 1:
      if (not unconverged and k > 0) or it >= maxits - 1:
        break
    inc it

  result.iterations = it + 1
  result.converged = not unconverged and k > 0
  result.labels = newSeq[int](n)
  if k == 0:
    for l in result.labels.mitems: l = -1
    result.dpsim = NaN
    result.expref = NaN
    result.netsim = NaN
    return

  var ex = newSeqOfCap[int](k)
  for i in 0 ..< n:
    if a[i, i] + r[i, i] > 0: ex.add i

  proc assign(s: Matrix, ex: seq[int], c: var seq[int]) =
    ## Each point to its most similar exemplar (the first on ties), each
    ## exemplar to itself.
    for i in 0 ..< s.n:
      var best = -Inf
      c[i] = 0
      for e, j in ex:
        if s[i, j] > best:
          best = s[i, j]
          c[i] = e
    for e, j in ex: c[j] = e

  var c = newSeq[int](n)
  assign(s, ex, c)
  # Refinement: in each cluster, the exemplar becomes the member with the
  # largest sum of similarities from the other members (and its preference).
  var members = newSeq[seq[int]](k)
  for i in 0 ..< n: members[c[i]].add i
  for e in 0 ..< k:
    var best = -Inf
    for j in members[e]:
      var t = 0.0
      for i in members[e]: t += s[i, j]
      if t > best:
        best = t
        ex[e] = j
  assign(s, ex, c)

  # Exemplars in increasing order, labels pointing into them.
  var isEx = newSeq[bool](n)
  for j in ex: isEx[j] = true
  var pos = newSeq[int](n)
  for j in 0 ..< n:
    if isEx[j]:
      pos[j] = result.exemplars.len
      result.exemplars.add j
  for i in 0 ..< n:
    let j = ex[c[i]]
    result.labels[i] = pos[j]
    if i == j: result.expref += s[i, i]
    else: result.dpsim += s[i, j]
  result.netsim = result.dpsim + result.expref

proc preferenceRange*(s: Matrix, exact = false): (float, float) =
  ## The preferences between which AP finds from 1 or 2 clusters (the lower
  ## bound) to n clusters (the upper bound), ignoring the diagonal of `s`.
  ## The lower bound is exact with `exact = true` (O(n^3)), otherwise it is a
  ## cheaper, smaller bound (O(n^2)), as in Frey and Dueck's preferenceRange.m.
  ## -Inf similarities are skipped in the sums.
  let n = s.n
  template sim(i, j: int): float = (if i == j: 0.0 else: s[i, j])
  template addFinite(acc: var float, v: float) =
    if v > -Inf:
      acc = (if acc == -Inf: v else: acc + v)
  var pmax = -Inf
  var dpsim1 = -Inf  # best net similarity with one exemplar
  for j in 0 ..< n:
    var t = -Inf
    for i in 0 ..< n:
      t.addFinite sim(i, j)
      if i != j and s[i, j] > pmax: pmax = s[i, j]
    if t > dpsim1: dpsim1 = t
  var pmin: float
  if dpsim1 == -Inf:
    pmin = NaN
  elif exact:
    var dpsim2 = -Inf  # best net similarity with two exemplars
    for j1 in 0 ..< n - 1:
      for j2 in j1 + 1 ..< n:
        var t = -Inf
        for i in 0 ..< n:
          t.addFinite max(sim(i, j1), sim(i, j2))
        if t > dpsim2: dpsim2 = t
    pmin = dpsim1 - dpsim2
  else:
    # Upper bound on dpsim2: every point to its most similar other point,
    # except the two for which that is least similar (the two exemplars).
    var total = -Inf
    var m1, m2 = Inf
    for i in 0 ..< n:
      var m = -Inf
      for j in 0 ..< n:
        if j != i and s[i, j] > m: m = s[i, j]
      if m > -Inf:
        total.addFinite m
        if m < m1:
          m2 = m1
          m1 = m
        elif m < m2:
          m2 = m
    pmin = if m2 == Inf: -Inf else: dpsim1 - total + m1 + m2
  (pmin, pmax)

{.pop.}
