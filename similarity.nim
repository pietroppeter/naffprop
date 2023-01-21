#[
  - SymMatrix is a triangular symmetric matrix with constant diagonal element (called preference)
  - a 3x3 SimMatrix should have 3 elements in data
  - a 4x4 SimMatrix should have 6 elements in data
  - a nxn SimMatrix should have n(n - 1)/2 elements
  - data.len should be kept equal to size*(size - 1)/2
  - elements are stored by row
  - for a 4x4 SimMatrix
    - index 0, 1 is element 0
    - index 0, 2 is element 1
    - index 0, 3 is element 2
    - index 1, 2 is element 3
    - and so forth:

      * 0 1 2
      * * 3 4
      * * * 5
      * * * *

  - initial row index for 4x4 matrix are: 0, 3, 5
  - initial row index for 5x5 matrix are: 0 (1 2 3), 4 (5 6), 7 (8), 9
  - initial row index for nxn matrix are: 0 (1 .. n -1), n, n + (n - 1), n + (n - 1) + (n - 2), ... (n-1)*n/2 - 1
    - for row index i I have i-th row index is n*i-T(i)? (where T(i) is i+1-th triangular number 1, 3, 6, 10, ... = (i+1)(i+2)/2)
]#
import std / [strformat, math]

type
  SimMatrix* = object
    n*: int
    pref*: float
    data*: seq[float]

type
  Point* = tuple
    x, y: float

func sim*(p, q: Point): float =
  - ((p.x - q.x)^2 + (p.y - q.y)^2) 

func expDataLen*(s: SimMatrix): int =
  (s.n*(s.n - 1)) div 2

template checkBounds =
  assert i >= 0 and i < s.n
  assert j >= 0 and j < s.n

template checkData =
  assert s.data.len == s.expDataLen

func index*(s: SimMatrix; i, j: int): int =
  checkBounds
  assert i != j
  if i < j:
    s.n*i - ((i + 1)*(i+2)) div 2 + j
  else:
    s.index(j, i)

func `[]`*(s: SimMatrix; i, j: int): float =
  checkBounds
  checkData
  if i == j:
    s.pref
  elif i < j:
    s.data[s.index(i, j)]
  else:
    s.data[s.index(j, i)]

iterator pairs*(s: SimMatrix): (int, int) =
  for i in 0 ..< s.n - 1:
    for j in (i + 1) ..< s.n:
      yield (i, j)

func `$`*(s: SimMatrix): string =
  for i in 0 ..< s.n:
    for j in 0 ..< s.n:
      result.add fmt"{s[i, j]:>7.2f}"
    result.add '\n'
  
func toSimMatrix*(points: seq[Point]): SimMatrix =
  result.n = len points
  result.data = newSeqUninitialized[float](result.expDataLen)
  for i, j in result:
    result.data[result.index(i, j)] = sim(points[i], points[j])

when isMainModule:
  import nimib
  nbInit

  block:  
    nbCode:
      let s = SimMatrix(n: 4, pref: 0.0, data: @[1.0, 2, 3, 4, 5, 6])
      for i, j in s:
        stdout.write s.index(i, j)
        stdout.write " "
        if j == s.n - 1:
          stdout.write '\n'
    nbCode:
      echo s
  
  nbCode:
    let points: seq[Point] = @[
      (-3.0, -1.0),
      (-2.0, -2.0),
      (-1.0, -3.0),
      (1.0, 2.0),
      (2.0, 1.0),
      (2.0, 2.0),
      (2.0, 3.0),
      (3.0, 2.0),
    ] 
    let s = toSimMatrix points
    echo s
  nbSave
