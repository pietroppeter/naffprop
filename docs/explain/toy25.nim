## The paper's 25 points (Frey and Dueck 2007, Fig. 1): the toy data set the
## explanation runs on. These are the original ToyProblemData.txt values from
## the authors' AP web page (psi.toronto.edu, now offline), as mirrored in
## https://github.com/jincheng9/AffinityPropagation (commit ad82f68, 2014).
## An affine fit onto the paper's Fig. 1A lands every point within 4 px.
## With negative squared distances and the median preference, AP picks points
## 2, 6 and 19 (from 0) as exemplars, as in the paper.

import data

const toy25Points* = [
  (-2.341500, 3.696800), (-1.109200, 3.111700), (-1.566900, 1.835100), (-2.658500, 0.664900),
  (-4.031700, 2.845700), (-3.081000, 2.101100), (2.588000, 1.781900), (3.292300, 3.058500),
  (4.031700, 1.622300), (3.081000, -0.611700), (0.264100, 0.398900), (1.320400, 2.207400),
  (0.193700, 3.643600), (1.954200, -0.505300), (1.637300, 1.409600), (-0.123200, -1.516000),
  (-1.355600, -3.058500), (0.017600, -4.016000), (1.003500, -3.590400), (0.017600, -2.420200),
  (-1.531700, -0.930900), (-1.144400, 0.505300), (0.616200, -1.516000), (1.707700, -2.207400),
  (2.095100, 3.430900)]

const paperExemplars* = [2, 6, 19]

func toy25*(): Ap2DInput =
  ## The points, with negative squared distances as similarities.
  var points: seq[Point]
  for (x, y) in toy25Points: points.add Point(x: x, y: y)
  initAp2DInput(points)
