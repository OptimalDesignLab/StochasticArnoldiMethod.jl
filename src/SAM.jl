
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
