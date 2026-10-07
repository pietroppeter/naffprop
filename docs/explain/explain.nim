## Types and data of the interactive explanation of affinity propagation (the
## paper's Fig. 1: points in the plane, the messages at every iteration).
##
## All the data is computed beforehand, in C, by naffprop's own `ap.nim` (its
## `onIteration` hook records the messages), and written as JSON; the page only
## reads it. The types compile to JS too, so the page parses the JSON with
## std/json's `to`.

import std/[algorithm, math, json]
import matrix, ap, rng

export matrix, ap

type
  Point* = object
    x*, y*: float

  Ap2DInput* = object
    ## Points in the plane and their similarities.
    seed*: int64                    ## of the generated points (0 if given)
    points*: seq[Point]
    similarityMatrix*: Matrix[float]  ## negative squared distances, 0 on the
                                      ## diagonal: the preference goes there
                                      ## only when AP runs

  ApParameters* = object
    damping*: float  ## lambda: each message is damping * old + (1 - damping) * new
    quantile*: float ## preference = this quantile of the similarities
                     ## between different points (0.5: the median, the paper's)
    maxits*: int     ## at most this many iterations
    convits*: int    ## stop when the exemplars stay the same this many iterations

  ApIterations* = object
    ## The messages after each iteration (from the first: before it, all are
    ## 0), and the decisions they imply at that time.
    r*, a*: seq[Matrix[float]]  ## responsibilities r(i, k), availabilities a(i, k)
    exemplars*: seq[seq[int]]   ## the points k with r(k, k) + a(k, k) > 0
    choice*: seq[seq[int]]      ## for each point i, the k maximizing
                                ## a(i, k) + r(i, k): i itself if it is an
                                ## exemplar, else the exemplar it picks

  ApRun* = object
    ## Everything a page needs about one run.
    input*: Ap2DInput
    params*: ApParameters
    preference*: float      ## the value the quantile gives, for every point
    iterations*: ApIterations
    result*: ApResult       ## naffprop's result: the exemplars after the last
                            ## iteration, refined (so they can differ from the
                            ## last `iterations.exemplars`, which are not)

func paperParameters*(): ApParameters =
  ## The paper's Fig. 1: damping 0.5, the median similarity as preference;
  ## maxits and convits as naffprop's (and R's) defaults.
  ApParameters(damping: 0.5, quantile: 0.5, maxits: 1000, convits: 100)

func negDistMatrix*(points: seq[Point]): Matrix[float] =
  ## Negative squared Euclidean distances, the similarity of the paper's Fig. 1.
  let n = points.len
  result = zeros[float](n)
  for i in 0 ..< n:
    for k in 0 ..< n:
      result[i, k] = -((points[i].x - points[k].x) ^ 2 + (points[i].y - points[k].y) ^ 2)

func initAp2DInput*(points: seq[Point], seed: int64 = 0): Ap2DInput =
  Ap2DInput(seed: seed, points: points, similarityMatrix: negDistMatrix(points))

func quantile*(xs: seq[float], q: float): float =
  ## As numpy's default (linear interpolation), so as naffprop's Python side.
  let v = sorted(xs)
  let h = q * float(v.len - 1)
  let lo = int(floor(h))
  let hi = min(lo + 1, v.len - 1)
  v[lo] + (h - float(lo)) * (v[hi] - v[lo])

func preferenceOf*(input: Ap2DInput, params: ApParameters): float =
  ## The quantile `params.quantile` of the similarities between different points.
  let s = input.similarityMatrix
  var xs: seq[float]
  for i in 0 ..< s.n:
    for k in 0 ..< s.n:
      if i != k: xs.add s[i, k]
  quantile(xs, params.quantile)

proc generatePoints*(seed: int64, centers: openArray[(float, float, int)],
                     sd: float): seq[Point] =
  ## Points around the centers (x, y, how many), gaussian with deviation `sd`,
  ## rounded to 2 decimals; the same for a seed on every platform (rng.nim).
  var g = initRng(seed)
  for (cx, cy, m) in centers:
    for _ in 0 ..< m:
      let x = round(cx + sd * g.gauss, 2)
      let y = round(cy + sd * g.gauss, 2)
      result.add Point(x: x, y: y)

proc toy25*(seed: int64 = 7): Ap2DInput =
  ## A stand-in for the paper's 25 points (ToyProblemData.txt, which we don't
  ## have): three groups in [-1, 1]².
  initAp2DInput(generatePoints(seed, [(-0.55, 0.45, 9), (0.5, 0.5, 8),
                                      (0.05, -0.5, 8)], 0.17), seed)

proc runAp*(input: Ap2DInput, params: ApParameters): ApRun =
  ## Runs naffprop's AP (no noise: the data is deterministic) and records the
  ## messages and decisions after every iteration.
  result = ApRun(input: input, params: params, preference: preferenceOf(input, params))
  var s = input.similarityMatrix
  let v = s.view
  v.prepare([result.preference], noise = false, seed = 0)
  var its: ApIterations
  proc record(it: int, r, a: Matrix[float], isEx: seq[bool]) =
    its.r.add r
    its.a.add a
    var ex: seq[int]
    var choice = newSeq[int](r.n)
    for i in 0 ..< r.n:
      if isEx[i]: ex.add i
      var best = -Inf
      for k in 0 ..< r.n:
        if a[i, k] + r[i, k] > best:
          best = a[i, k] + r[i, k]
          choice[i] = k
    its.exemplars.add ex
    its.choice.add choice
  result.result = v.affinityPropagation(params.maxits, params.convits,
                                        params.damping, record)
  result.iterations = its

func roundSignificant*(x: float, digits: int): float =
  ## x to `digits` significant digits (the messages don't need more on a page,
  ## and the JSON is much smaller).
  if x == 0 or x.isNaN or x.classify in {fcInf, fcNegInf}: return x
  let e = floor(log10(abs(x)))
  let f = pow(10.0, float(digits - 1) - e)
  round(x * f) / f

proc roundMessages*(run: var ApRun, digits = 4) =
  for m in run.iterations.r.mitems:
    for x in m.data.mitems: x = roundSignificant(x, digits)
  for m in run.iterations.a.mitems:
    for x in m.data.mitems: x = roundSignificant(x, digits)
