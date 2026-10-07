# The explanation imports naffprop's own Nim code (ap.nim, matrix.nim, rng.nim).
import std/os
switch("path", thisDir() / ".." / ".." / "src" / "naffprop")
# As in core.nims: no fused multiply-add, so the data is the same on every platform.
switch("passC", "-ffp-contract=off")
# nimib and karax (for the pages): from NIMIB_DEPS, which build.sh fills with
# pinned commits.
let deps = getEnv("NIMIB_DEPS")
if deps.len > 0:
  for p in ["nimib/src", "karax", "fusion/src", "nim-markdown/src", "parsetoml/src", "jsony/src"]:
    switch("path", deps / p)
