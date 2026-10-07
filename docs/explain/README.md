# Interactive explanation of affinity propagation (work in progress)

The goal: the explanation of the original paper (Frey and Dueck, Science 2007, Fig. 1)
made interactive, written in Nim with [nimib](https://github.com/pietroppeter/nimib) and
Karax. This first part is the data: everything the pages show is computed beforehand by
naffprop's own `ap.nim` and written as JSON, which the pages read.

- `explain.nim`: the types (`Point`, `Ap2DInput`, `ApParameters`, `ApIterations`,
  `ApRun`, plus naffprop's `ApResult`) and how the data is made. The types also compile to
  JS, so a page parses the JSON with std/json's `to`.
- `gen_data.nim`: writes `data/toy25.json` (not committed, about 1 MB, 80 KB gzipped):
  `nim c -r docs/explain/gen_data.nim`.
- `test_explain.nim`: `nim c -r docs/explain/test_explain.nim`, and after `gen_data`, for
  the JS side, `nim js -d:nodejs -r docs/explain/test_explain.nim`.

The data: 25 points standing in for the paper's (its ToyProblemData.txt is no longer
online), similarity the negative squared distance, the paper's parameters (damping 0.5,
preference the median similarity), no noise. The messages are rounded to 4 significant
digits.
