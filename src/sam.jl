@doc """
### StochasticArnoldiMethod.sam

The Stochastic Arnoldi Method (SAM) approximately minimizes functions that may
have inaccurate data

**Inputs**

* `func`: function to be minimized; functor for function and gradient information
* `x0`: the initial design
* `tol`: the tolerance target
* `options` (optional): dictionary of options

**Returns**

* `x`: approximate local minimizer 
* `hist`: history data structure

"""->
function sam(func::Function, x0, tol,
             options::Dict{ASCIIString,Any}=default_options)
  merge(default_options, options)
  checkSAMOptions(options)
  @assert( size(x0,1) == length(x0), 
           "x0 should be a column vector" )
  @assert( tol >= eps(1.0),
           "tol should be a positive number greater than eps" )

  # create arrays to store sampled data, and sample at x0
  xdata = zeros(n, options["num_sample"])
  fdata = zeros(1, options["num_sample"])
  gdata = zeros(n, options["num_sample"])
  xdata[:,1] = x0
  func(view(xdata,:,1), view(fdata,:,1), view(gdata,:,1))

  # generate Arnoldi sample
  m = options["num_sample"]-1
  eigenvals = zeros(m)
  eigenvecs = zeros(m,m)
  grad_red = zeros(m)
  sample_size = arnoldiSample(func, xdata, fdata, gdata, options["alpha"],
                              options["num_sample"], eigenvals, eigenvecs,
                              grad_red)

  # initialize history data structure  
  hist = SAMHistory(func, sample_size+1, fdata[1,1], norm(gdata[:,1]),
                    options["truth"])

  # start outer, nonlinear iterations
  radius = options["init_radius"]
  for k = 1:options["max_iter"]
    
    # check for convergence and display convergence
    if options["grad_method"] == "average"
      grad_norm = norm(mean(gdata,2))
    elseif options["grad_method"] == "dirderiv"
      grad_norm = norm(grad_red)
    end
    if k == 1
      grad_norm0 = grad_norm
    end
    if options["display"]
      println("iter = ",k,": rel. grad norm = ",grad_norm/grad_norm0)
      println("\tradius = ",radius)
    end
    if (grad_norm < grad_norm0*tol)
      break
    end

    if options["grad_method"] == "average"
      # use gradient averaging for the reduced gradient
      [dx_red, pred, pos_def] = trust(V.'*mean(gdata,2), diag(L), radius)
      dx = V*dx_red
      pred = -pred
      #pred = -mean(gdata,2).'*V*dx_red - 0.5*dx_red.'*diag(L)*dx_red
      x = mean(xdata,2) + dx
    elseif options["grad_method"] == "dirderiv"
      # use the directional-derivative for the reduced gradient
      [dx_red, pred, pos_def] = trust(gred, diag(L), radius)
      dx = V*dx_red
      pred = -pred
      #pred = -gred.'*dx_red - 0.5*dx_red.'*diag(L)*dx_red
      x = xdata(:,1) + dx
    end
  
end

@doc """
### StochasticArnoldiMethod.checkOptions

Check the SAM option values in the given dictionary.  Throws an assertion error
if any of the options are invalid.

**Inputs**

* `options`: the options dictionary to be checked.

"""->
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
  @assert( options["init_radus"] >= options["min_radius"] &&
           options["init_radius"] <= options["max_radius"],
           "option init_radius should be between min_radius and max_radius" )

end

@doc """
### StochasticArnoldiMethod.updateSAMHistory

Updates the fields in the given SAMHistory datatype

**Inputs**

* `count`: number of function evaluations used in most recent iteration
* `val`: current estimate of the function value
* `grad`: current estimate of the gradient norm

**In/Outs**

* `hist`: SAMHistory convergence history datatype

"""->
function updateSAMHistory(count::Int, val::Float64, grad::Float64,   
                          hist::SAMHistory)
  push!(hist.func_count, hist.func_count[end] + count)
  push!(hist.func_val, val)
  push!(hist.grad_norm, grad)
end

function SAM(fun, x0, hess_rank, num_sample, tol, variant)
  #TODO: need to pass function somehow

  if (variant < 0) || (variant > 1)
    error("variant option must be 0 or 1")
  end

  n = size(x0, 1)
  MaxIter = 10
  radius = norm(x0)
  max_radius = 10.0*radius
  min_radius = 0.0001;
  model_radius = 0.5
  x = x0

  # Initial samples
  xdata = zeros(n, num_sample)
  fdata = zeros(1, num_sample)
  gdata = zeros(n, num_sample)
  xdata[:,1] = zeros(n,1)
  (fdata[1,1], gdata[:,1]) = fun(x)

  # Estimate the Hessian
  [xdata, fdata, gdata, V, L, new_rank, gred] =
    ApproxArnoldi(fun, x, fdata[1,1], gdata[:,1], 
                  model_radius, hess_rank, num_sample - 1)

  hist = Dict()
  hist["fnccount"] = zeros(MaxIter+1,1)
  hist["fval"] = zeros(MaxIter+1,1)
  hist["norm"] = zeros(MaxIter+1,1)
  hist["norm0"] = 0.0
  hist["iters"] = 0

  hist["fnccount"][1] = num_sample
  (hist["fval"][1], gex) = fun(x, 'no_noise')

  # start outer iterations
  for k = 1:MaxIter

    # check for convergence
    if variant == 0
      grad_norm = norm(mean(gdata,2))
    elseif variant == 1
      grad_norm = norm(gred)
    end
    hist["norm"][k] = grad_norm
    if k == 1
      grad_norm0 = grad_norm
      hist["norm0"] = grad_norm0
    end
    if (mod(k,1) == 0)
      println("iter = ", k,": rel grad norm = ", grad_norm/grad_norm0)
      println("  radius = ", radius)
    end
    if (grad_norm < grad_norm0*tol)
      break
    end
    hist["iters"] = k+1

    xold = x

    if variant == 0

      # step-averaging (used for CSE 2015)
      # TODO: check this diag(L)
      (dx_red, pred, pos_def) = trust(V.' * mean(gdata,2), diag(L), radius)
      dx = V*dx_red
      pred = -pred
      # line 71 in SAM.m not put in here
      x = mean(xdata,2) + dx
    elseif variant == 1
      # directional derivative
      (dx_red, pred, pos_def) = trust(gred, diag(L), radius)
      dx = V*dx_red
      pred = -pred
      # line 78 in SAM.m not put in here
      x = xdata[:,1] + dx
    end
    
    (f, g) = fun(x)
    ared = fdata[1,1] - f
    rho = ared/pred
    # rho = 1.0 # TEMP: no globalization
    println("  pred = ", pred, ": ared = ", ared, ": rho = ", rho)
    if rho < 1e-3
      radius = 0.25*radius 
      # radius = max(0.25*radius, min_radius)
    else
      if ( (rho > 0.75) && (abs(norm(dx) - radius) < 1e-4) )
        radius = min(2*radius, max_radius)
      end
    end
    count = 0
    if (rho > 1e-4)
      # keep new solution
      fdata[1,1] = f
      gdata[:,1] = g
    else
      # revert
      x = xold
      (f, g) = fun(x)
      fdata[1,1] = f
      gdata[:,1] = g
      count = 1
    end
    
    # estimate eigenvectors of Hessian
    [xdata, fdata, gdata, V, L, new_rank, gred] = 
        ApproxArnoldi(fun, x, fdata(1,1), gdata(:,1), model_radius, 
          hess_rank, num_sample-1)

    hist["fnccount"][k+1] = hist["fnccount"][k] + num_sample-1 + count
    (hist["fval"][k+1], gex) = fun(x, 'no_noise')   # need to evaluate w/o noise
  end






  return x, hist

end
