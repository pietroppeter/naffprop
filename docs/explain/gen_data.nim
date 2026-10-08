## Writes the data of the explanation: nim c -r docs/explain/gen_data.nim
## (from the repository root) -> docs/explain/data/toy25.json.
import std/[json, os]
import data, toy25

let dir = currentSourcePath().parentDir / "data"
createDir dir
var run = runAp(toy25(), fig1Parameters())
run.roundMessages(4)
writeFile(dir / "toy25.json", $(%run))
echo "toy25: ", run.result.iterations, " iterations, exemplars ",
     run.iterations.exemplars[^1], " (refined: ", run.result.exemplars, ")"
var first: seq[int]  # the iteration (from 1) when k + 1 exemplars appear
for it, ex in run.iterations.exemplars:
  while first.len < ex.len: first.add it + 1
echo "exemplar count first reached at iterations ", first
