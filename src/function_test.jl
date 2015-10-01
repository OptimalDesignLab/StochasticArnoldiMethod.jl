module FunctionTest
# to run:
#   include("function_test.jl")
#   using .FunctionTest
#   main_fun(1,2)

export main_fun

function main_fun(a,b)

  c = inner_fun(a,b)
  d = c+a
  return d

end

function inner_fun(a,b)
  c = a+b
  return c

end

end
