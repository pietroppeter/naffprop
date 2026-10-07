# The explanation imports naffprop's own Nim code (ap.nim, matrix.nim, rng.nim).
import std/os
switch("path", thisDir() / ".." / ".." / "src" / "naffprop")
# As in core.nims: no fused multiply-add, so the data is the same on every platform.
switch("passC", "-ffp-contract=off")
