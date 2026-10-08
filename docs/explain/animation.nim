## The paper's Fig. 1A as an animation: the points of an `ApRun`,
## colored by the evidence that each is an exemplar, with an arrow from i to k
## as dark as i's belief that k is its exemplar, and under them a color bar of
## the evidence with every point's current value; play, step or scrub through
## the iterations. A Karax app (JS only) mounted on a div of the page:
## `figureHtml(id, params)` writes the div and its caption at build time,
## `mountAnimation(id, run)` starts the app in the browser.
##
## The evidence colors are fixed (they read on light and dark backgrounds);
## the rest come from CSS custom properties (--ap-ink, --ap-frame, --muted),
## so the page decides them, for both themes.

import std/[math, strutils]
import explain
when defined(js):
  include karax/prelude
  import karax/[kdom, vstyles]
  {.warning[CStringConv]: off.}  # attribute values are cstrings in JS

func f(x: float): string = formatFloat(x, ffDecimal, 2)

type Frame = object
  ## The square of data coordinates shown, and its size in SVG units.
  x0, y0, side, px: float

func frameOf(points: seq[Point], px: float): Frame =
  var lo = Inf
  var hi = -Inf
  for p in points:
    lo = min(lo, min(p.x, p.y))
    hi = max(hi, max(p.x, p.y))
  let pad = 0.08 * (hi - lo)
  Frame(x0: lo - pad, y0: lo - pad, side: hi - lo + 2 * pad, px: px)

func sx(fr: Frame, x: float): float = (x - fr.x0) / fr.side * fr.px
func sy(fr: Frame, y: float): float = fr.px - (y - fr.y0) / fr.side * fr.px

# Evidence colors. Negative evidence follows viridis, from blue (the lowest
# evidence of the run) through green to yellow (just below 0): a perceptually
# uniform scale with no red in it. Positive evidence, an exemplar, is red, from
# a light red at 0 to a deep red at the highest evidence of the run.
const viridis = [(59, 82, 139), (44, 114, 142), (33, 145, 140), (39, 173, 129),
                 (94, 201, 98), (170, 220, 50), (253, 231, 37)]  # 0.25 .. 1
const redLow = (255, 107, 94)
const redHigh = (178, 24, 24)

func mix(a, b: (int, int, int), t: float): string =
  func c(x, y: int): int = int(round(float(x) + (float(y) - float(x)) * t))
  "rgb(" & $c(a[0], b[0]) & "," & $c(a[1], b[1]) & "," & $c(a[2], b[2]) & ")"

func evidenceColor*(e, lo, hi: float): string =
  ## The color of evidence `e` on a run whose evidence spans lo < 0 < hi.
  if e > 0:
    mix(redLow, redHigh, clamp(e / hi, 0.0, 1.0))
  else:
    let x = clamp(1 - e / lo, 0.0, 1.0) * float(viridis.high)
    let i = min(int(x), viridis.high - 1)
    mix(viridis[i], viridis[i + 1], x - float(i))

func evidence(r, a: Matrix[float], k: int): float = r[k, k] + a[k, k]

func beliefs*(r, a: Matrix[float], i: int, temperature: float): seq[float] =
  ## How strongly point i believes each k is its exemplar, summing to 1: the
  ## messages are log-probabilities (up to a constant), so a softmax of
  ## a(i, k) + r(i, k) on the scale `temperature`. Before any message, every
  ## candidate gets 1/n; at convergence, the chosen exemplar gets almost 1.
  let n = r.n
  result = newSeq[float](n)
  var best = -Inf
  for k in 0 ..< n: best = max(best, a[i, k] + r[i, k])
  var t = 0.0
  for k in 0 ..< n:
    result[k] = exp((a[i, k] + r[i, k] - best) / temperature)
    t += result[k]
  for k in 0 ..< n: result[k] /= t

# The figure's frame and caption, in HTML at build time: the legend, then
# the parameters of the run.
const evidenceDot* = """<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><defs><linearGradient id="ap-key-grad"><stop offset="0" stop-color="rgb(59,82,139)"/><stop offset="0.5" stop-color="rgb(253,231,37)"/><stop offset="1" stop-color="rgb(178,24,24)"/></linearGradient></defs><circle cx="8" cy="8" r="7" fill="url(#ap-key-grad)"/></svg>"""
const arrow* = """<svg width="26" height="10" viewBox="0 0 26 10" aria-hidden="true"><line x1="1" y1="5" x2="21" y2="5" style="stroke:var(--ap-ink)" stroke-width="1.5"/><path d="M19,1 L25,5 L19,9 z" style="fill:var(--ap-ink)"/></svg>"""

func legendHtml*(params: ApParameters): string =
  ## What color and arrows mean, then the parameters of the run.
  let pref = if params.quantile == 0.5: "median similarity"
             else: "similarity quantile " & $params.quantile
  "<div class=\"key\">" &
    "<span>" & evidenceDot & "</span><span>evidence r(k,k) + a(k,k) that k is an " &
    "exemplar (it is one when positive)</span>" &
    "<span>" & arrow & "</span><span>i → k: i's belief that k is its exemplar</span>" &
    "</div>\n" &
    "<p class=\"params\"><strong>parameters</strong> <span>preference " & pref & "</span> · <span>damping λ = " &
    $params.damping & "</span> · <span>convits " & $params.convits &
    "</span> · <span>maxits " & $params.maxits & "</span></p>"

func figureHtml*(id: string, params: ApParameters): string =
  ## The frame an animation mounts in (the div `id`), with its legend.
  "<figure>\n<div id=\"" & id & "\"></div>\n<figcaption>\n" & legendHtml(params) &
    "\n</figcaption>\n</figure>"

when defined(js):
  proc mountAnimation*(rootId: cstring, run: ApRun, px = 400.0, msPerStep = 90) =
    let pts = run.input.points
    let n = pts.len
    let fr = frameOf(pts, px)
    let last = run.iterations.r.len   # 0 is before any message, then 1 .. last
    let temperature = abs(run.preference) / 10
    let zero = zeros[float](n)
    let rad = 0.02 * px
    # the evidence scale is the range of the whole run, so a color means the
    # same at every iteration
    var lo = Inf
    var hi = -Inf
    for t in 1 .. last:
      for k in 0 ..< n:
        let e = evidence(run.iterations.r[t - 1], run.iterations.a[t - 1], k)
        lo = min(lo, e)
        hi = max(hi, e)
    # keep 0 strictly inside the scale, even for a run with no exemplar
    lo = min(lo, -1e-9)
    hi = max(hi, 1e-9)
    # for how many iterations, up to t, each point has been an exemplar: points
    # become opaque as they settle, over `convits` iterations
    var streak = newSeq[seq[int]](last + 1)
    streak[0] = newSeq[int](n)
    for t in 1 .. last:
      streak[t] = streak[t - 1]
      for k in 0 ..< n:
        if k in run.iterations.exemplars[t - 1]: inc streak[t][k]
        else: streak[t][k] = 0
    let settle = max(run.params.convits, 1)
    let barH = 0.13 * px  # the color bar under the points
    var t = 0
    var playing = false
    var timer: Interval
    var kxi: KaraxInstance  # our own instance: several animations can share a page

    proc stop() =
      if playing:
        clearInterval(timer)
        playing = false

    proc go(j: int) =
      stop()
      t = clamp(j, 0, last)

    proc tick() =
      if t < last: inc t
      else: stop()
      redraw(kxi)

    proc toggle() =
      if playing: stop()
      else:
        if t >= last: t = 0
        playing = true
        timer = setInterval(tick, msPerStep)

    proc picture(): VNode =
      let r = if t == 0: zero else: run.iterations.r[t - 1]
      let a = if t == 0: zero else: run.iterations.a[t - 1]
      let exemplars = if t == 0: newSeq[int]() else: run.iterations.exemplars[t - 1]
      result = buildHtml(svg(viewBox = "0 0 " & f(px) & " " & f(px), class = "ap-plot",
                             role = "img")):
        defs:
          marker(id = "ap-head", viewBox = "0 0 10 10", refX = "9", refY = "5",
                 markerWidth = "5", markerHeight = "5", orient = "auto-start-reverse"):
            path(d = "M0,0 L10,5 L0,10 z", class = "ap-head")
        rect(width = f(px), height = f(px), class = "ap-frame")
        for i in 0 ..< n:
          let b = beliefs(r, a, i, temperature)
          for k in 0 ..< n:
            if k != i and b[k] >= 0.02:
              let (x1, y1) = (sx(fr, pts[i].x), sy(fr, pts[i].y))
              let (x2, y2) = (sx(fr, pts[k].x), sy(fr, pts[k].y))
              let d = max(hypot(x2 - x1, y2 - y1), 1e-9)
              let stopAt = rad * (if k in exemplars: 1.9 else: 1.4)
              line(x1 = f(x1), y1 = f(y1), x2 = f(x2 - (x2 - x1) / d * stopAt),
                   y2 = f(y2 - (y2 - y1) / d * stopAt), class = "ap-arrow",
                   opacity = f(b[k]), `stroke-width` = f(0.6 + 1.4 * b[k]),
                   `marker-end` = "url(#ap-head)")
        for k in 0 ..< n:
          let e = evidence(r, a, k)
          let isEx = k in exemplars
          # before any message there is no evidence yet: a neutral point.
          # Points are see-through until they become exemplars, then turn
          # opaque as they stay exemplars (settled after `convits` iterations)
          let color = if t == 0: "var(--muted)" else: evidenceColor(e, lo, hi)
          let opacity = 0.35 + 0.65 * min(streak[t][k] / settle, 1.0)
          circle(cx = f(sx(fr, pts[k].x)), cy = f(sy(fr, pts[k].y)),
                 r = f(if isEx: 1.5 * rad else: rad),
                 class = (if isEx: "ap-point ap-exemplar" else: "ap-point"),
                 fill = color, stroke = color, `fill-opacity` = f(opacity)):
            title: text "point " & $k & ": evidence r(k,k) + a(k,k) = " &
                        formatFloat(e, ffDecimal, 3)

    proc colorBar(): VNode =
      ## The evidence scale, linear from the lowest to the highest of the run,
      ## with 0 marked and a tick for every point now. The bar stretches to the
      ## figure's width; its labels are HTML, so they keep their size.
      let r = if t == 0: zero else: run.iterations.r[t - 1]
      let a = if t == 0: zero else: run.iterations.a[t - 1]
      const (w, h, steps) = (1000.0, 24.0, 60)
      proc bx(e: float): float = w * (e - lo) / (hi - lo)
      result = buildHtml(tdiv(class = "ap-scale")):
        tdiv(class = "ap-scale-row"):
          span: text "← not an exemplar"
          span: text "exemplar →"
        svg(viewBox = "0 0 " & f(w) & " " & f(h), preserveAspectRatio = "none",
            class = "ap-bar", role = "img", `aria-label` = "Evidence color scale"):
          for j in 0 ..< steps:
            # one rect per step, colored at its middle
            let e = lo + (hi - lo) * (j.float + 0.5) / steps.float
            rect(x = f(j.float * w / steps.float), y = "6", width = f(w / steps.float + 1),
                 height = "12", fill = evidenceColor(e, lo, hi))
          for x in [0.5, bx(0), w - 0.5]:
            line(x1 = f(x), y1 = "0", x2 = f(x), y2 = f(h), class = "ap-mark")
          if t > 0:
            for k in 0 ..< n:
              let x = bx(evidence(r, a, k))
              line(x1 = f(x), y1 = "2", x2 = f(x), y2 = "22", class = "ap-tick")
        tdiv(class = "ap-scale-zero"):
          span(style = style(StyleAttr.left, f(100 * bx(0) / w) & "%")): text "0"

    proc createDom(): VNode =
      let exemplars = if t == 0: newSeq[int]() else: run.iterations.exemplars[t - 1]
      let label = if t == 0: "Before any message"
                  elif t == last and run.result.converged: "Iteration " & $t & ", converged"
                  else: "Iteration " & $t
      let playLabel = if playing: "Pause" else: "Play"
      result = buildHtml(tdiv(class = "ap-anim")):
        picture()
        colorBar()
        tdiv(class = "ap-controls"):
          button(class = "ap-play", onclick = toggle): text playLabel
          button(class = "ap-step", title = "Previous iteration",
                 onclick = proc() = go(t - 1)): text "‹"
          button(class = "ap-step", title = "Next iteration",
                 onclick = proc() = go(t + 1)): text "›"
          input(`type` = "range", id = "ap-slider", min = "0", max = $last,
                value = $t, class = "ap-slider", `aria-label` = "Iteration"):
            proc oninput(ev: Event; n: VNode) = go(parseInt($n.value))
        tdiv(class = "ap-readout"):
          span(class = "ap-it"): text label
          span(class = "ap-ex"):
            text $exemplars.len & (if exemplars.len == 1: " exemplar" else: " exemplars")

    kxi = setRenderer(createDom, root = rootId)
    redraw(kxi)

  proc enableArrowKeys*() =
    ## Left and right arrow keys step the page's first animation, except while
    ## typing in an input (the slider handles its own arrows).
    document.addEventListener("keydown", proc (ev: Event) =
      let e = KeyboardEvent(ev)
      if ev.target != nil and Element(ev.target).nodeName == "INPUT": return
      let steps = document.querySelectorAll(".ap-step")
      if steps.len < 2: return
      if e.key == "ArrowLeft":
        steps[0].click()
        ev.preventDefault()
      elif e.key == "ArrowRight":
        steps[1].click()
        ev.preventDefault())
