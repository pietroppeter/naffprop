## Affinity propagation with ScaleAP's pruning (Shiokawa, AAAI 2021): the
## same messages as `ap.affinityPropagation`, without storing most of them.
##
## ScaleAP observes that most messages need not be computed one by one:
##
## - A responsibility r(i, k) where k has never been the best candidate of i
##   (the argmax of a(i, k) + s(i, k)) follows the same update as every other
##   such pair of row i, so r(i, k) = (1 - lam^t) s(i, k) - b(i), with b(i) a
##   number per row.
## - An availability a(i, k) where r(i, k) has never been positive follows the
##   same update as every other such pair of column k, so a(i, k) = a(k), a
##   number per column.
##
## So only the pairs that have been a best candidate or had a positive
## responsibility (a few per row: the close neighbours) store their messages.
## Memory is the similarities plus O(n) plus those pairs, instead of three
## n x n matrices, and each iteration reads the similarities once.
##
## Unlike ScaleAP's paper and reference code, the updates are those of Frey
## and Dueck (the self-responsibility uses the availabilities, the
## availabilities use the new responsibilities), so the exemplars are those of
## `ap.affinityPropagation` (up to rounding).

import std/algorithm
import ap

type
  Pair[T] = object
    ## A pair (i, col) whose messages are stored.
    r, a: T
    col: int32

{.push checks: off.}  # hot loops: indices are in range by construction

proc affinityPropagationPruned*[T](s: MatrixView[T], maxits = 1000, convits = 100,
                                   damping = 0.9): ApResult =
  ## Affinity propagation on the similarities `s` (preferences on the
  ## diagonal, prepared by `ap.prepare`), as `ap.affinityPropagation`.
  let n = s.n
  let lam = damping
  var
    pairs = newSeq[seq[Pair[T]]](n)  # stored pairs of each row, by column
    b = newSeq[float](n)             # r(i, k) = (1 - lam^t) s(i, k) - b(i)
    acol = newSeq[float](n)          # a(i, k) = acol(k)
    rd = newSeq[float](n)            # self-responsibilities r(i, i)
    ad = newSeq[float](n)            # self-availabilities a(i, i)
    maxOut = newSeq[float](n)        # largest s(i, k), k != i, of the pairs not stored
    colsum, colsumPrev = newSeq[float](n)
    saved: seq[float]
    added: seq[int]
    conv = initConvergence(n, convits)
    isEx = newSeq[bool](n)
    lt = 1.0                         # lam^t
    it = 0

  proc updateMaxOut(i: int) =
    var m = -Inf
    var p = 0
    for k in 0 ..< n:
      while p < pairs[i].len and pairs[i][p].col < k: inc p
      if k != i and (p == pairs[i].len or pairs[i][p].col != k) and float(s[i, k]) > m:
        m = float(s[i, k])
    maxOut[i] = m

  proc insert(i, k: int, r, a: float) =
    let q = pairs[i].lowerBound(k, proc (x: Pair[T], k: int): int = cmp(int(x.col), k))
    pairs[i].insert(Pair[T](r: T(r), a: T(a), col: int32(k)), q)

  for i in 0 ..< n: updateMaxOut(i)
  while true:
    let ltPrev = lt
    lt *= lam
    for k in 0 ..< n: colsum[k] = 0
    for i in 0 ..< n:
      # The stored availabilities of the previous iteration, which needed the
      # column sums of all the rows.
      if it > 0:
        for x in pairs[i].mitems:
          let v = float(x.r)
          x.a = T((1 - lam) * min(0.0, colsumPrev[x.col] - max(v, 0.0)) + lam * float(x.a))
      # max1, max2 and argmax of a(i, k) + s(i, k), with the stored
      # availabilities put in acol for this row.
      saved.setLen pairs[i].len + 1
      for q, x in pairs[i]:
        saved[q] = acol[x.col]
        acol[x.col] = float(x.a)
      saved[^1] = acol[i]
      acol[i] = ad[i]
      var max1, max2 = -Inf
      var kmax = 0
      for k in 0 ..< n:
        let v = acol[k] + float(s[i, k])
        if v > max1:
          max2 = max1
          max1 = v
          kmax = k
        elif v > max2:
          max2 = v
      for q, x in pairs[i]: acol[x.col] = saved[q]
      acol[i] = saved[^1]
      # The best candidate is stored from now on: its update differs.
      var changed = false
      if kmax != i:
        let q = pairs[i].lowerBound(kmax, proc (x: Pair[T], k: int): int = cmp(int(x.col), k))
        if q == pairs[i].len or pairs[i][q].col != kmax:
          insert(i, kmax, (1 - ltPrev) * float(s[i, kmax]) - b[i], acol[kmax])
          changed = true
      # Responsibilities.
      for x in pairs[i].mitems:
        let m = if x.col == kmax: max2 else: max1
        x.r = T((1 - lam) * (float(s[i, x.col]) - m) + lam * float(x.r))
      rd[i] = (1 - lam) * (float(s[i, i]) - (if kmax == i: max2 else: max1)) + lam * rd[i]
      b[i] = lam * b[i] + (1 - lam) * max1
      # Pairs not stored whose responsibility turns positive are stored.
      let th = b[i] / (1 - lt)
      if maxOut[i] > th:
        added.setLen 0
        var p = 0
        for k in 0 ..< n:
          while p < pairs[i].len and pairs[i][p].col < k: inc p
          if k != i and float(s[i, k]) > th and (p == pairs[i].len or pairs[i][p].col != k):
            added.add k
        for k in added:
          insert(i, k, (1 - lt) * float(s[i, k]) - b[i], acol[k])
        changed = true
      if changed: updateMaxOut(i)
      for x in pairs[i]:
        if x.r > 0: colsum[x.col] += float(x.r)
      colsum[i] += rd[i]
    # Availabilities: the shared one of each column and the self ones; the
    # stored ones at the next iteration.
    for k in 0 ..< n:
      acol[k] = (1 - lam) * min(0.0, colsum[k]) + lam * acol[k]
      ad[k] = (1 - lam) * (colsum[k] - rd[k]) + lam * ad[k]
      isEx[k] = ad[k] + rd[k] > 0
    swap colsum, colsumPrev
    if conv.update(it, isEx, maxits): break
    inc it

  result = finish(s, conv, it, isEx)

{.pop.}
