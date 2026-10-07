# Build options for core.nim, read by nim whoever invokes it (nimlang's hatch
# hook runs `nim c`). No fused multiply-add contraction: the C compiler would
# otherwise fuse `a*b + c` on aarch64 but not on x86_64, so the messages would
# round differently and results could differ between platforms.
switch("passC", "-ffp-contract=off")
