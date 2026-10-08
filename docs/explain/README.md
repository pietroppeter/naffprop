# Interactive explanation of affinity propagation (work in progress)

The goal: the explanation of the original paper (Frey and Dueck, Science 2007, Fig. 1,
[pdf](https://people.csail.mit.edu/kjhsiao/Frey2007.pdf))
made interactive, written in Nim with [nimib](https://github.com/pietroppeter/nimib) and
Karax. Everything the pages show is computed beforehand by naffprop's own `ap.nim` and
written as JSON, which the pages embed and read.

- `explain.nim`: the types (`Point`, `Ap2DInput`, `ApParameters`, `ApIterations`,
  `ApRun`, plus naffprop's `ApResult`) and how the data is made. The types also compile to
  JS, so a page parses the JSON with std/json's `to`.
- `gen_data.nim`: writes `data/toy25.json` (not committed, about 270 KB):
  `nim c -r docs/explain/gen_data.nim`.
- `animation.nim`: the Fig. 1A animation, a Karax app (JS), and its figure and legend
  (HTML at build time).
- `theme.nim`, `style.css`: the look of the pages, and `nbHeader`.
- `fig1a.nim`: the page, "How exemplars emerge". It needs nimib (>= 0.4.2) and karax, from
  `NIMIB_DEPS` (see `config.nims`).
- `build.sh`: fetches nimib and karax at pinned commits, generates the data, runs the
  tests and builds the pages into `site/` (not committed): `sh docs/explain/build.sh`.
  CI runs it on every push and publishes `site/` to GitHub Pages from main.
- `test_explain.nim`: `nim c -r docs/explain/test_explain.nim`, and after `gen_data`, for
  the JS side, `nim js -d:nodejs -r docs/explain/test_explain.nim`.

- `toy25.nim`: the paper's 25 points, the authors' ToyProblemData.txt (no longer online at
  its source; from a mirror, see the file). AP finds the paper's exemplars, 2, 6 and 19.

The data: the paper's 25 points, similarity the negative squared distance, preference the median similarity,
damping 0.9 and convits 10 (`fig1Parameters`: the exemplars emerge over more iterations
than with the paper's damping 0.5), no noise. The messages are rounded to 4 significant
digits.
