## The paper's 25 points (Frey and Dueck 2007, Fig. 1): the toy data set the
## explanation runs on. An approximate reconstruction, digitized from the
## paper's figure (the original ToyProblemData.txt is no longer online). In the
## paper, AP picks points 2, 6 and 19 (from 0) as exemplars.

import explain

const toy25Points* = [
  (0.25, 9.027), (2.05, 9.784), (2.85, 7.730), (1.45, 8.108), (0.15, 7.081),
  (1.50, 6.541), (7.55, 7.676), (6.25, 8.865), (6.90, 9.351), (8.15, 9.081),
  (3.25, 6.054), (9.20, 8.486), (1.85, 8.811), (8.90, 7.351), (8.10, 6.703),
  (3.20, 3.892), (4.40, 4.108), (5.65, 3.946), (5.95, 2.432), (4.85, 2.649),
  (4.90, 1.784), (3.30, 8.919), (6.15, 1.135), (3.40, 1.568), (9.35, 6.811)]

const paperExemplars* = [2, 6, 19]

func toy25*(): Ap2DInput =
  ## The points, with negative squared distances as similarities.
  var points: seq[Point]
  for (x, y) in toy25Points: points.add Point(x: x, y: y)
  initAp2DInput(points)
