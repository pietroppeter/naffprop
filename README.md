# affprop

[Affinity Propagation] in [Nim] for Python (and Nim) - WIP

Project started during [![Open Source Saturday](https://img.shields.io/badge/%E2%9D%A4%EF%B8%8F-open%20source%20saturday-F64060.svg)](https://www.meetup.com/it-IT/Open-Source-Saturday-Milano/)

The idea is to implement the [Affinity Propagation] clustering algorithm in Nim and make it available in Python through [nimporter](https://github.com/Pebaz/nimporter). I want to see how much friction there is to support this use case for nim (basically Nim as a Cython alternative). The advantage of Affinity Propagation over K-means is that the number of cluster is automatically detected (although there is a knob that can be used to increase it or decrease it). The disadvantage is that it is memory and computation intensive.

Ideally I would like to experiment some of the "scalable" variants, one possible target and a good reference is ["Scalable Affinity Propagation for massive datasets", Shiokawa 2021](https://scholar.google.it/scholar?hl=it&as_sdt=2005&sciodt=0%2C5&cites=17534306066151546043&scipsc=&q=Scalable+affinity+propagation+for+massive+datasets&btnG=).
This seems to solve the issue of computational complexity (goes from O(N^2T) to O(NT)) but it still
seems to have a memory complexity of O(N^2) (and in particular complexity of similarity computation is still O(N^2)).

For the implementation I would like to have an implementation that can also work on Javascript backend, so that I can make interactive visualizations. This is why I am also implementing some basic Matrix objects instead of using libraries. The obvious performant option would be to use [arraymancer] and maybe later I will implement it. It would not be bad to try and use a `MatrixLike` [concept], although I have practically zero experience there.

todo:
- [x] similarity matrix type (did not really need this but it was fun)
- [x] matrix type
- [ ] complete ap algorithm
- [ ] visualizations (with p5nim?)
- [ ] python packaging with nimporter

[Affinity Propagation]: https://en.wikipedia.org/wiki/Affinity_propagation
[arraymancer]: https://github.com/mratsim/Arraymancer
[concept]: https://nim-lang.org/docs/manual_experimental.html#concepts
[Nim]: https://nim-lang.org
