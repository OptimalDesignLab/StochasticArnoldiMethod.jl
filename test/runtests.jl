using StochasticArnoldiMethod
using ArrayViews
using FactCheck

include("test_arnoldi_sample.jl")
include("test_trust.jl")
include("test_sam.jl")

FactCheck.exitstatus()