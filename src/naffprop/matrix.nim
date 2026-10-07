## A dense square matrix of floats, stored by row.
##
## This is all the linear algebra affinity propagation needs: its messages are
## elementwise updates plus row maxima and column sums, so there is no BLAS
## and no dependency (it also keeps the door open for Nim's JS backend).

type
  Matrix* = object
    n*: int
    data*: seq[float]

func zeros*(n: int): Matrix =
  Matrix(n: n, data: newSeq[float](n * n))

template `[]`*(m: Matrix; i, j: int): float =
  m.data[i * m.n + j]

template `[]=`*(m: var Matrix; i, j: int; v: float) =
  m.data[i * m.n + j] = v

func `$`*(m: Matrix): string =
  for i in 0 ..< m.n:
    for j in 0 ..< m.n:
      result.add $m[i, j]
      result.add(if j == m.n - 1: '\n' else: ' ')
