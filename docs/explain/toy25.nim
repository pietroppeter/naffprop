## The paper's 25 points (Frey and Dueck 2007, Fig. 1): the toy data set the
## explanation runs on. Recovered by Pietro (the paper's ToyProblemData.txt is
## no longer online); we have no citable source for them.

import explain

const toy25Points* = [
  (0.12, 0.74), (0.25, 0.82), (0.18, 0.65), (0.29, 0.71), (0.08, 0.85),
  (0.68, 0.15), (0.75, 0.28), (0.62, 0.22), (0.81, 0.11), (0.70, 0.33),
  (0.85, 0.88), (0.92, 0.75), (0.78, 0.82), (0.89, 0.68), (0.95, 0.81),
  (0.45, 0.48), (0.52, 0.53), (0.41, 0.55), (0.48, 0.42), (0.38, 0.46),
  (0.15, 0.18), (0.22, 0.25), (0.11, 0.30), (0.28, 0.12), (0.20, 0.22)]

func toy25*(): Ap2DInput =
  ## The points, with negative squared distances as similarities.
  var points: seq[Point]
  for (x, y) in toy25Points: points.add Point(x: x, y: y)
  initAp2DInput(points)
