# Build options for core.nim, read by nim whoever invokes it (nimlang's hatch
# hook runs `nim c`). No fused multiply-add contraction: the C compiler would
# otherwise fuse `a*b + c` on aarch64 but not on x86_64, so the messages would
# round differently and results could differ between platforms.
switch("passC", "-ffp-contract=off")

# NAFFPROP_STD_RANDOM=1 at build time (only "1"): draw the noise with std/random (see rng.nim).
if getEnv("NAFFPROP_STD_RANDOM") == "1":
  switch("define", "naffpropStdRandom")
