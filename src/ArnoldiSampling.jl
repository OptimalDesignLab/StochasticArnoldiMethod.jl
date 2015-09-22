module ArnoldiSampling

include("trust.jl")

using .Trust

# aa = hello()

# println("aa = ",aa)

aa,bb = trust(61,62,63)

println(aa)
println(bb)

end
