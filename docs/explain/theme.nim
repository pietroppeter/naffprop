## The look of the explanation pages: nimib's default theme with our own
## stylesheet (style.css) inlined, no highlight.js (code is highlighted at build
## time), so nothing loads from a CDN but the fonts; and nbHeader, the page's
## title block.

import std/json
import nimib

const fonts = """<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400&display=swap">"""

const style = "<style>\n" & staticRead("style.css") & "</style>"

func noHeader(blk: JsonNode, nb: Nb): string = ""

# The page's title, with an eyebrow above it (the line that says what the page
# belongs to); it replaces nimib's header bar.
newNbBlock(NbHeader):
  title: string
  eyebrow: string
  toHtml:
    "<header class=\"nb-header\">" &
      (if blk.eyebrow.len > 0: "<p class=\"eyebrow\">" & blk.eyebrow & "</p>" else: "") &
      "<h1>" & blk.title & "</h1></header>"

proc header*(nb: var Nb, title: string, eyebrow = "") =
  nb.add newNbHeader(title = title, eyebrow = eyebrow)
  nb.title = title

template nbHeader*(title: string, eyebrow = "") = nb.header(title, eyebrow)

proc useExplainTheme*(nb: var Nb) =
  ## Call right after `nbInit`.
  nb.doc.context["stylesheet"] = %(fonts & "\n" & style)
  nb.doc.context["highlight"] = %""
  nb.doc.context["nb_style"] = %""
  nb.doc.context["disableHighlightJs"] = %true
  nb.backend.partials["header"] = noHeader
