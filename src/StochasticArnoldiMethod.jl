__precompile__(false)
module StochasticArnoldiMethod

using ArrayViews

export SAM
export arnoldiSample
export trust
export secular
export default_options
export SAMHistory

@doc """
StochasticArnoldiMethod.default_options

**Options**

* `alpha`: step size used in Arnoldi sampling (default = 1e-2)
* `num_sample`: maximum number of samples in Arnoldi (default = 10)
* `grad_method`: gradient approximation method, \"average\" or \"dirderiv\"
                 (default = \"average\")
* `max_iter`: maximum number of nonlinear iterations (default = 10)
* `max_radius`: maximum trust-region radius (default = 10.0)
* `min_radius`: minimum trust-region radius (default = 1e-4)
* `init_radius`: initial trust-region radius (defalut = 1.0)
* `truth`: if true, the user function can be evaluated exactly for testing
           (default = false)
* `display_level`: indicates how much information to display to stout
                   0: only errors are dislayed (default)
                   1: major iteration information
                   2: major and minor iteration information

"""->
default_options =
  Dict{ASCIIString,Any}(
                        "alpha"=>1e-2,
                        "num_sample"=>10,
                        "grad_method"=>"average",
                        "max_iter"=>10,
                        "max_radius"=>10.0,
                        "min_radius"=>1e-4,
                        "init_radius"=>1.0,
                        "truth"=>false,
                        "display_level"=>0
                        )   
@doc """
StochasticArnoldiMethod.SAMHistory

Type to record the convergence history of SAM

**Fields**

* `func_count`: cumulative number of calls to the user function at each iteration
* `func_val`: function value at each iteration
* `grad_norm`: estimated gradient norm at each iteration

"""->
type SAMHistory
  func_count::Array{Int,1}
  func_val::Array{Float64,1}
  grad_norm::Array{Float64,1}

  function SAMHistory(func::Function, count::Int, val::Float64, grad::Float64;
                      exact::Bool=false)
    func_count = zeros(Int, (1))
    func_count[1] = count
    func_val = zeros(1)
    grad_norm = zeros(1)
    if exact
      # use exact values for the history, if available
      temp_val = zeros(1,1)
      temp_grad = zeros(n,1)
      func(view(xdata,:,1), temp_val, temp_grad, exact=exact)
      hist.func_val[1] = temp_val[1,1]
      hist.grad_norm[1] = norm(temp_grad)
    else
      func_val[1] = val
      grad_norm[1] = grad
    end
    new(func_count, func_val, grad_norm)
  end
end

include("arnoldi_sampling.jl")
include("trust.jl")
include("sam.jl")

end