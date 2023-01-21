import std / [math, sequtils, strformat]

type
  Matrix* = object
    n*: int
    data*: seq[float]

template checkBounds =
  assert i >= 0 and i < m.n
  assert j >= 0 and j < m.n

template checkData =
  assert m.data.len == m.n ^ 2

func `[]`*(m: Matrix; i, j: int): float =
  checkBounds
  checkData
  m.data[i*m.n + j]

proc `[]=`*(m: var Matrix; i, j: int, v: float) =
  checkBounds
  checkData
  m.data[i*m.n + j] = v

proc zeros*(n: int): Matrix =
  result.n = n
  result.data = newSeqWith(n^2, 0.0)

func `$`*(m: Matrix): string =
  for i in 0 ..< m.n:
    for j in 0 ..< m.n:
      result.add fmt"{m[i, j]:>7.2f}"
    result.add '\n'
  
when isMainModule:
  import nimib
  nbInit
  nbCode:
    var m = zeros(4)
    m[1, 2] = 5.0
    m[3, 0] = 4.0
    echo m
  nbSave