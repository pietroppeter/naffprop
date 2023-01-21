import tabby

type
  Point* = object
    x*, y*: float

var data = "data.csv".readFile.fromCsv(seq[Point], hasHeader = false, separator = " ")
echo data.len
echo data[0]