import datamancer

let df = readCsv("data.csv", sep = ' ', colNames = @["x", "y"])

var data: seq[tuple[x, y: float]]
for row in df:
  data.add (row["x"].toFloat, row["y"].toFloat)

import jsony

writeFile("data.json", data.toJson)
