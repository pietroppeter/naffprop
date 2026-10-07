## The random numbers of naffprop: standard normal draws from a seeded generator,
## for the tiny noise that breaks ties in the similarities.
##
## By default a small generator of our own (splitmix64 and Box-Muller): std/random
## imports std/sysrand, which links macOS's Security framework, and that framework
## is not available when nimlang cross-compiles for macOS. It also gives the same
## draws for a seed on every platform.
##
## Compile with `-d:naffpropStdRandom` to use std/random instead (for a wheel,
## build with the NAFFPROP_STD_RANDOM environment variable set: see core.nims).
## The draws then differ from the default generator's, so the same seed gives
## different noise, and cross-building for macOS needs a macOS SDK (SDKROOT).

when defined(naffpropStdRandom):
  import std/random

  type Rng* = Rand

  proc initRng*(seed: int64): Rng = initRand(seed)

  proc gauss*(r: var Rng): float = random.gauss(r)

else:
  import std/math

  type Rng* = object
    ## splitmix64: a small seeded generator.
    state: uint64

  proc initRng*(seed: int64): Rng = Rng(state: cast[uint64](seed))

  proc next(r: var Rng): uint64 =
    r.state += 0x9E3779B97F4A7C15'u64
    var z = r.state
    z = (z xor (z shr 30)) * 0xBF58476D1CE4E5B9'u64
    z = (z xor (z shr 27)) * 0x94D049BB133111EB'u64
    z xor (z shr 31)

  proc uniform(r: var Rng): float =
    ## In (0, 1]: 53 random bits.
    float((r.next shr 11) + 1) * pow(2.0, -53)

  proc gauss*(r: var Rng): float =
    ## Standard normal, by the Box-Muller transform.
    sqrt(-2 * ln(r.uniform)) * cos(2 * PI * r.uniform)
