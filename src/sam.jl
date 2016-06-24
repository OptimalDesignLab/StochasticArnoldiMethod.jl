"""
### StochasticArnoldiMethod.SAM

The Stochastic Arnoldi Method (SAM) approximately minimizes functions that may
have inaccurate data

**Inputs**

* `func`: function to be minimized; provides function and gradient information
* `x0`: the initial design
* `tol`: the tolerance target
* `options` (optional): dictionary of options

**Returns**

* `x`: approximate local minimizer
* `f`: objective function at `x`
* `hist`: history data structure

"""
function SAM(func::Function, x0, tol,
             options::Dict{ASCIIString,Any}=default_options)
  merge(default_options, options)
  checkSAMOptions(options)
  @assert( size(x0,1) == length(x0), 
           "x0 should be a column vector" )
  @assert( tol >= eps(1.0),
           "tol should be a positive number greater than eps" )

  # create arrays to store sampled data, and sample at x0
  n = size(x0,1)
  x = deepcopy(x0)
  xdata = zeros(n, options["num_sample"])
  fdata = zeros(1, options["num_sample"])
  gdata = zeros(n, options["num_sample"])
  xdata[:,1] = x0
  func(view(xdata,:,1), view(fdata,:,1), view(gdata,:,1))

  # generate Arnoldi sample
  m = options["num_sample"]-1
  eigenvals = zeros(m)
  eigenvecs = zeros(n,m)
  grad_red = zeros(m)
  alpha = options["alpha"]
  sample_size, err_est = arnoldiSample(func, xdata, fdata, gdata, alpha,
                                       options["num_sample"], eigenvals,
                                       eigenvecs, grad_red)

  # initialize history data structure
  hist = SAMHistory(func, sample_size, fdata[1,1], norm(gdata[:,1]),
                    exact=options["truth"])

  # start outer, nonlinear iterations
  radius = options["init_radius"]
  local grad_norm0::Float64
  dg_sum = 0.0
  for k = 1:options["max_iter"]
    
    # check for convergence and display convergence
    if options["grad_method"] == "average"
      grad_norm = norm(gdata[:,1]) #norm(mean(gdata[:,1:sample_size+1],2))
    elseif options["grad_method"] == "dirderiv"
      grad_norm = norm(grad_red)
    end
    if k == 1
      grad_norm0 = grad_norm
    end
    if options["display_level"] > 0
      @printf("\niter = %d: obj = %g: optimality = %g: radius = %g\n",
              k-1, fdata[1,1], grad_norm/grad_norm0, radius)

      # estimate the noise level
      #gmean = mean(gdata[:,1:sample_size+1],2)
      #dg = gdata[:,1:sample_size+1] - gmean*ones(1,sample_size+1)
      #dg_sum += sum(dg.*dg)
      #gstd = sqrt(sum(sum(dg.*dg))./(n*sample_size))
      #println("\n\testimate of gradient error = ",sqrt(dg_sum./(n*hist.func_count[end])))
      #println("\n\tsmallest eigenvalue = ",minimum(eigenvals[1:sample_size]))

    end
    if (grad_norm < grad_norm0*tol)
      break
    end

    B = diagm(eigenvals[1:sample_size])
    local trust_active::Bool
    local pred::Float64
    if options["grad_method"] == "average"
      # use gradient averaging for the reduced gradient
      g = vec(eigenvecs[:,1:sample_size].'*mean(gdata[:,1:sample_size+1],2))
      p, pred, trust_active = trust(g, B, radius, display_level=
                                    options["display_level"])
      x = mean(xdata[:,1:sample_size+1],2) + eigenvecs[:,1:sample_size]*p
    elseif options["grad_method"] == "dirderiv"
      # use the directional-derivative for the reduced gradient
      g = grad_red
      p, pred, trust_active = trust(g, B, radius, display_level=
                                    options["display_level"])
      x = xdata[:,1] + eigenvecs[:,1:sample_size]*p
    end

    # evaluate at new point, and globalize if necessary
    f_new = zeros((1,1))
    g_new = zeros((n,1))
    func(x, view(f_new,:,1), view(g_new,:,1))
    ared = fdata[1,1] - f_new[1,1]
    rho = ared/pred
    if options["display_level"] > 0 
      @printf("\tpred = %g: ared = %g: rho = %g\n", pred, ared, rho)
    end
    if rho < eps()
      radius = max(0.25*radius, options["min_radius"])
    else
      if rho > 0.75 && rho < 1.25 && trust_active
        radius = min(2.*radius, options["max_radius"])
      end
    end
    if rho > eps()
      # some reduction, so keep the new solution
      xdata[:,1] = x[:]
      fdata[1,1] = f_new[1,1]
      gdata[:,1] = g_new[:,1]
    end
    
    # run Arnoldi sampling, update history and output
    sample_size, err_est = arnoldiSample(func, xdata, fdata, gdata, alpha,
                                options["num_sample"], eigenvals, eigenvecs,
                                grad_red)
    updateSAMHistory(func, sample_size, fdata[1,1], norm(gdata[:,1]), hist,
                     exact=options["truth"])
  end
  x = xdata[:,1]
  return x, fdata[1,1], hist
end

"""
### StochasticArnoldiMethod.checkOptions

Check the SAM option values in the given dictionary.  Throws an assertion error
if any of the options are invalid.

**Inputs**

* `options`: the options dictionary to be checked.

"""
function checkSAMOptions(options::Dict{ASCIIString,Any})
  @assert( options["alpha"] >= eps(1.0),
           "option alpha should be a positive number greater than eps" )
  @assert( options["num_sample"] >= 1,
           "option num_sample must be greater than or equal to 1" )
  @assert( options["grad_method"] == "average" || 
           options["grad_method"] == "dirderiv",
           "option grad_method must be \"average\" or \"dirderiv\"" )
  @assert( options["max_iter"] >= 1,
           "option max_iter must be greater than or equal to 1" )
  @assert( options["max_radius"] >= eps(1.0),
           "option max_radius should be a positive number greater than eps" )
  @assert( options["min_radius"] >= eps(1.0),
           "option min_radius should be a positive number greater than eps" )
  @assert( options["max_radius"] >= options["min_radius"],
           "option max_radius should greater than option min_radius" )
  @assert( options["init_radius"] >= eps(1.0),
           "option init_radius should be a positive number greater than eps" )
  @assert( options["init_radius"] >= options["min_radius"] &&
           options["init_radius"] <= options["max_radius"],
           "option init_radius should be between min_radius and max_radius" )
  @assert( options["display_level"] >= 0 &&
           options["display_level"] <= 2,
           "option display_level must be 0, 1, or 2" )
end

"""
### StochasticArnoldiMethod.updateSAMHistory

Updates the fields in the given SAMHistory datatype

**Inputs**

* `func`: function to be minimized; provides function and gradient information
* `count`: number of function evaluations used in most recent iteration
* `val`: current estimate of the function value
* `grad`: current estimate of the gradient norm

**In/Outs**

* `hist`: SAMHistory convergence history datatype

"""
function updateSAMHistory(func::Function, count::Int, val::Float64,
                          grad::Float64, hist::SAMHistory; exact::Bool=false)
  push!(hist.func_count, hist.func_count[end] + count)
  if exact
    # use exact values for the history, if available, for testing
    temp_val = zeros(1,1)
    temp_grad = zeros(n,1)
    func(view(xdata,:,1), temp_val, temp_grad, exact=exact)
    push!(hist.func_val, temp_val[1,1])
    push!(hist.grad_norm, temp_grad[:,1])
  else
    push!(hist.func_val, val)
    push!(hist.grad_norm, grad)
  end
end

