
# How to use (in REPL):
#   include("function_anon_test.jl")
#   f = x->x^2
#   test_anon(f,2)

function test_anon(func, n)
  a = func(n)
  a = a+1
  return a
end
