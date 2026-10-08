## Dense square matrices of floats, stored by row.
##
## This is all the linear algebra affinity propagation needs: its messages are
## elementwise updates plus row maxima and column sums, so there is no BLAS
## and no dependency (it also keeps the door open for Nim's JS backend).
##
## A `Matrix` owns its elements. A `MatrixView` points to elements stored
## elsewhere: in a `Matrix`, or in a numpy array, so that the similarities can
## be used where Python stored them, without a copy.

when defined(js):
  # No UncheckedArray in JS: a view points to the seq of a Matrix instead.
  template `[]`*[T](e: ptr seq[T]; k: int): T = e[][k]
  template `[]=`*[T](e: ptr seq[T]; k: int; v: T) = e[][k] = v

type
  Matrix*[T: SomeFloat] = object
    n*: int
    data*: seq[T]

  MatrixView*[T: SomeFloat] = object
    n*: int
    data*: (when defined(js): ptr seq[T] else: ptr UncheckedArray[T])

func zeros*[T](n: int): Matrix[T] =
  Matrix[T](n: n, data: newSeq[T](n * n))

func view*[T](m: var Matrix[T]): MatrixView[T] =
  ## A view of `m`, valid while `m` is alive and not resized.
  when defined(js): MatrixView[T](n: m.n, data: m.data.addr)
  else: MatrixView[T](n: m.n, data: cast[ptr UncheckedArray[T]](m.data[0].addr))

template `[]`*[T](m: Matrix[T] | MatrixView[T]; i, j: int): T =
  m.data[i * m.n + j]

template `[]=`*[T](m: var Matrix[T]; i, j: int; v: T) =
  m.data[i * m.n + j] = v

template `[]=`*[T](m: MatrixView[T]; i, j: int; v: T) =
  m.data[i * m.n + j] = v

func `$`*[T](m: Matrix[T] | MatrixView[T]): string =
  for i in 0 ..< m.n:
    for j in 0 ..< m.n:
      result.add $m[i, j]
      result.add(if j == m.n - 1: '\n' else: ' ')
