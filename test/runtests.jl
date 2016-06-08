using StochasticArnoldiMethod
using ArrayViews
using FactCheck

#include("test_arnoldi_sampling.jl")
#include("test_trust.jl")
include("test_sam.jl")

FactCheck.exitstatus()