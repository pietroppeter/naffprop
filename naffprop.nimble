# The Nim package: the algorithm in src/naffprop.nim and src/naffprop/*.nim, plain Nim
# with no dependency. Only .nim files are installed: core.nim (the nimpy bindings for the
# Python package) comes along but is never imported by the Nim package.
# The version follows pyproject.toml's (tests/test_naffprop.py checks it), and both
# are released by the same vX.Y.Z tag.

version       = "0.3.0"
author        = "Pietro Peterlongo"
description   = "Affinity propagation clustering (Frey and Dueck 2007) with R's apcluster features"
license       = "MIT"
srcDir        = "src"
installExt    = @["nim"]

requires "nim >= 2.2.0"
