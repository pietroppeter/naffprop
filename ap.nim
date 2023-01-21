import similarity, matrix

func affinityPropagation*(s: SimMatrix, iters = 100, lambda = 0.5): seq[int] =
  var
    r = zeros(s.n)
    a = zeros(s.n)
    rho = zeros(s.n)
    alpha = zeros(s.n)
  
