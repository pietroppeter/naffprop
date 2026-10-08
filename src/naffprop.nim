## Affinity propagation in Nim: `import naffprop` for everything, or one module,
## e.g. `import naffprop/ap` for the dense algorithm only.
##
## - `naffprop/matrix`: `Matrix` and `MatrixView`, dense square matrices
## - `naffprop/ap`: affinity propagation on a dense matrix, `preferenceRange`
## - `naffprop/sparse`: on the stored pairs of a sparse matrix
## - `naffprop/leveraged`: leveraged AP, on the similarities to a sample of the points
## - `naffprop/scaleap`: ScaleAP's pruning of the dense loop
##
## Plain Nim with no dependency, for the C and JS backends (sparse, leveraged
## and scaleap are C only). The Python package wraps the same code
## (`naffprop/core.nim`, not part of the Nim package).

import naffprop/[matrix, ap, sparse, scaleap]
export matrix, ap, sparse, scaleap
when not defined(js):
  # its similarities are a pointer to memory stored elsewhere (by numpy)
  import naffprop/leveraged
  export leveraged
