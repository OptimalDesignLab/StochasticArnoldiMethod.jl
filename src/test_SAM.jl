include("SAM.jl")

function test_SAM()

  x0 = [4;2]
  hess_rank = 2
  num_sample = 10
  tol = 1e6
  variant = 0
  (x, hist) = SAM(test_fun1, x0, hess_rank, num_sample, tol, variant)

end

function test_fun1(x)

  f = x1^2 + x2^2
  g = [2*x1; 2*x2]
end







