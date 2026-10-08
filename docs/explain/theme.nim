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

const githubLogo = """<svg aria-hidden="true" width="1.25em" height="1.25em" viewBox="0 0 16 16"><path fill="currentColor" fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59c.4.07.55-.17.55-.38c0-.19-.01-.82-.01-1.49c-2.01.37-2.53-.49-2.69-.94c-.09-.23-.48-.94-.82-1.13c-.28-.15-.68-.52-.01-.53c.63-.01 1.08.58 1.23.82c.72 1.21 1.87.87 2.33.66c.07-.52.28-.87.51-1.07c-1.78-.2-3.64-.89-3.64-3.95c0-.87.31-1.59.82-2.15c-.08-.2-.36-1.02.08-2.12c0 0 .67-.21 2.2.82c.64-.18 1.32-.27 2-.27c.68 0 1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82c.44 1.1.16 1.92.08 2.12c.51.56.82 1.27.82 2.15c0 3.07-1.87 3.75-3.65 3.95c.29.25.54.73.54 1.48c0 1.07-.01 1.93-.01 2.2c0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path></svg>"""

# The page's title, with an eyebrow above it (the line that says what the page
# belongs to) and, next to the eyebrow, a GitHub link to the repository; it
# replaces nimib's header bar.
newNbBlock(NbHeader):
  title: string
  eyebrow: string
  repo: string
  toHtml:
    "<header class=\"nb-header\"><div class=\"top\">" &
      "<p class=\"eyebrow\">" & blk.eyebrow & "</p>" &
      (if blk.repo.len > 0: "<a class=\"repo\" href=\"https://github.com/" & blk.repo &
        "\" title=\"source on GitHub\">" & githubLogo & "<span>" & blk.repo & "</span></a>" else: "") &
      "</div><h1>" & blk.title & "</h1></header>"

proc header*(nb: var Nb, title: string, eyebrow = "", repo = "") =
  nb.add newNbHeader(title = title, eyebrow = eyebrow, repo = repo)
  nb.title = title

template nbHeader*(title: string, eyebrow = "", repo = "") = nb.header(title, eyebrow, repo)

proc useExplainTheme*(nb: var Nb) =
  ## Call right after `nbInit`.
  nb.doc.context["stylesheet"] = %(fonts & "\n" & style)
  nb.doc.context["highlight"] = %""
  nb.doc.context["nb_style"] = %""
  nb.doc.context["disableHighlightJs"] = %true
  nb.backend.partials["header"] = noHeader
