## Writes the data of the explanation: nim c -r docs/explain/gen_data.nim
## (from the repository root) -> docs/explain/data/toy25.json.
import std/[json, os]
import explain

let dir = currentSourcePath().parentDir / "data"
createDir dir
var run = runAp(toy25(), paperParameters())
run.roundMessages(4)
writeFile(dir / "toy25.json", $(%run))
echo "toy25: ", run.result.iterations, " iterations, exemplars ",
     run.iterations.exemplars[^1], " (refined: ", run.result.exemplars, ")"
