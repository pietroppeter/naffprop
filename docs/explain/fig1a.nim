## How exemplars emerge: the paper's Fig. 1A, one iteration at a time.
## Build: sh docs/explain/build.sh (or, after gen_data.nim, nim c -r fig1a.nim from
## docs/explain) -> docs/explain/fig1a.html
import std/[json, strutils, xmltree]
import nimib
import nimib/[highlight, renders]
import theme, explain, animation

nbInit
nb.useExplainTheme()

func detailsFilePartial(blk: JsonNode, nb: Nb): string =
  ## nbFile as a collapsed <details>: the file's name and length as summary,
  ## then the code, escaped, and highlighted at build time if it is Nim.
  let name = blk{"filename"}.getStr.replace("../../", "")
  let content = blk{"content"}.getStr
  let ext = blk{"ext"}.getStr
  let code = if ext == "nim": content.highlightNim else: xmltree.escape(content)
  "<details class=\"nb-file\"><summary>" & xmltree.escape(name) & " <span>· " &
    $content.countLines & " lines</span></summary>\n" &
    preCodeTag(ext, code, highlight = false) & "\n</details>"

nb.backend.partials["nbFile"] = detailsFilePartial

nbHeader("How exemplars emerge", eyebrow = "naffprop · explaining affinity propagation")

nbText: """
Affinity propagation (Frey and Dueck, [*Science* 2007](https://doi.org/10.1126/science.1136800),
[pdf](https://people.csail.mit.edu/kjhsiao/Frey2007.pdf)) lets every point exchange two
kinds of messages with every other point until a few of them stand out as **exemplars**.
This is the paper's Fig. 1A, one iteration at a time: press Play, step with the arrows
(or the arrow keys), or drag the slider.
"""

# The animation runs in the browser on data computed beforehand by naffprop's own
# ap.nim (gen_data.nim); the JSON is embedded in the page at build time.
let apRoot = "ap-fig1a"
let run = parseJson(readFile("data/toy25.json")).to(ApRun)
nbRawHtml figureHtml(apRoot, run.params)
nbJsFromCodeOwnFile(apRoot):
  import std/json
  import explain, animation
  const data = staticRead("data/toy25.json")
  mountAnimation(apRoot, parseJson(data).to(ApRun))
  enableArrowKeys()

nbText: """
## The code

From the algorithm to the page. The source of this page itself is under *Show Source*
at the bottom.

**The algorithm.** naffprop's own affinity propagation, the code that runs when you
call it from Python. Each iteration updates the responsibilities r, then the
availabilities a; a point is an exemplar while r(k,k) + a(k,k) > 0. The `onIteration`
hook, compiled only when given, is how this page records the messages.
"""
nbFile("../../src/naffprop/ap.nim")

nbText: """
**The data.** The types of the explanation (points, parameters, the messages at every
iteration) and the run behind the figure: the paper's own 25 points (`toy25.nim`),
similarities -‖xᵢ − xₖ‖², and AP with the hook recording every iteration. It runs beforehand, in C; the page reads the
result as JSON.
"""
nbFile("explain.nim")
nbFile("toy25.nim")

nbText: """
**The animation.** A Karax app compiled to JavaScript: from the recorded messages it
draws the points, the arrows and the color bar of one iteration, and the controls move
through them. The figure around it and its legend are plain HTML, written at build time.
"""
nbFile("animation.nim")

nbText: """
**The page.** The theme plugs our stylesheet into nimib, with no highlight.js, and adds
`nbHeader`, the title with the line above it; the stylesheet has the colors for light
and dark.
"""
nbFile("theme.nim")
nbFile("style.css")
nbSave
