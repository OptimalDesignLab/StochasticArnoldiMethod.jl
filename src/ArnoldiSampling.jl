module ArnoldiSampling

include("trust.jl")

using .Trust

# aa = hello()

# println("aa = ",aa)

A = zeros(3,3)

A[1,1] = 3.931544008059447
A[1,2] = -4.622828930484834
A[2,1] = A[1,2]
A[1,3] = 1.571893108754884
A[3,1] = A[1,3]
A[2,2] = 5.438436601890520
A[2,3] = -1.853920290644159
A[3,2] = A[2,3]
A[3,3] = -1.853920290644159

b = [-0.964888535199277; -0.157613081677548; -0.970592781760616];

radius = 1e6


# (aa,bb) = trust(61,62,63)
(s, val, posdef, count, lambda) = trust(b, A, radius)

# println(aa)
# println(bb)

x = s
pred = val
lam = lambda

println(x)
println(pred)

end
