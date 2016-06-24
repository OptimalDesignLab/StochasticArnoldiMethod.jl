"""
### StochasticArnoldiMethod.trust

Solves a trust-region constrained quadratic optimization problem; suitable for
relatively small problems with explicit (dense) Hessians.

**Inputs**

* `g`: gradient of the objective
* `B`: Hessian of objective, or an approximation to it
* `Δ`: trust-radius value

**Returns**

* `p`: the constrained step
* `pred`: the predicted reduction in the objective
* `active`: true if the trust-region constraint is active, false otherwise

"""
function trust(g::Array{Float64,1}, B::Array{Float64,2}, Δ::Float64;
               display_level::Int=0)
  @assert( norm(B - B.') < 100.*eps(), "B must be a symmetric matrix")
  @assert( size(B,1) == size(B,2) == size(g,1),
           "B must be a square matrix and/or the same size as g")
  @assert( Δ > 0 , "Δ must be a positive number" )
  n = size(g,1)
  
  B_eig = eigvals(B)
  eigmin = minimum(B_eig)
  λ = 0.0
  if eigmin > eps(1.0)
    # Hessian is semi-definite, so solve for p and check if ||p|| is in trust
    # region radius
    p, f, df = secular(g, B, λ, Δ, display_level=display_level)
    if f < 0.0 # i.e. ||p|| < Δ
      # compute predicted decrease in the objective and return
      pred = -dot(p, 0.5*B*p + g)
      return p, pred, false
    end
    if display_level > 0
      println("\ttrust: norm(p) = ",norm(p)," > ",Δ)
    end
  end
  # if we get here, either the Hessian is negative-definite or ||p|| > radius

  # bracket the Lagrange multiplier lambda
  max_brk = 20
  dλ = 0.1*max(-eigmin, eps())
  λ_h = max(-eigmin, 0.0) + dλ
  p, f_h, df = secular(g, B, λ_h, Δ, display_level=display_level)
  for k = 1:max_brk
    if display_level > 1
      @printf("\ttrust: (λ_h, f_h) = (%g,%g)\n", λ_h, f_h)
    end
    if f_h > 0.0
      break
    end
    dλ *= 0.1
    λ_h = max(-eigmin, 0.0) + dλ
    p, f_h, df = secular(g, B, λ_h, Δ, display_level=display_level)
  end
  
  dλ = sqrt(eps())
  λ_l = max(-eigmin, 0.0) + dλ
  p, f_l, df = secular(g, B, λ_l, Δ, display_level=display_level)
  for k = 1:max_brk
    if display_level > 1
      @printf("\ttrust: (λ_l, f_l) = (%g,%g)\n", λ_l, f_l)
    end
    if f_l < 0.0
      break
    end
    dλ *= 100.0
    λ_l = max(-eigmin, 0.0) + dλ
    p, f_l, df = secular(g, B, λ_l, Δ, display_level=display_level)
  end

  λ = 0.5*(λ_l + λ_h)
  if display_level > 1
    @printf("\ttrust: initial lambda = %g\n",λ)
  end
  
  # Apply (safe-guarded) Newton's method to find lambda
  max_Newt = 50
  dλ_old = abs(λ_h - λ_l)
  dλ = dλ_old
  tol = sqrt(eps())
  lam_tol = tol*dλ
  p, f, df = secular(g, B, λ, Δ, display_level=display_level)
  res0 = abs(f)
  l = 0
  for l = 1:max_Newt
    if display_level > 1
      @printf("\ttrust: Newton iter = %d: res = %g: lambda = %g\n", l, abs(f), λ)
      #println("\ttrust: Newton iter = ",l,": res = ",abs(f),": lambda = ",λ)
    end
    # check if p lies on the trust region, if so exit
    if abs(f) < tol*res0 || abs(dλ) < lam_tol
      if display_level > 0
        @printf("\ttrust: Newton converged with lambda = %g\n", λ)
      end
      break
    end
    
    # choose safe-guarded step
    if ((λ - λ_h)*df - f)*((λ - λ_l)*df - f) > 0.0 || abs(2.*f) > abs(dλ_old*df)
      # use bisection if Newton step is out of range or not decreasing fast
      dλ_old = dλ
      dλ = 0.5*(λ_h - λ_l)
      λ = λ_l + dλ
      if λ_l == λ
        break
      end
    else
      # Newton step is acceptable
      dλ_old = dλ
      dλ = f/df
      temp = λ
      λ -= dλ
      if temp == λ
        break
      end
    end
      
    # evaluate secular function at new lambda
    p, f, df = secular(g, B, λ, Δ, display_level=display_level)
    f < 0.0 ? λ_l = λ : λ_h = λ
  end
  if l == max_Newt
    error("trust: Newton's method failed to converge in ",max_Newt," iterations")
  end
  
  # compute predicted decrease in the objective and return
  pred = -dot(p, 0.5*B*p + g)  
  return p, pred, true
end

"""
### StochasticArnoldiMethod.secular

Computes the secular function for trust-region optimization

**Inputs**

* `g`: gradient of the objective
* `B`: Hessian of objective, or an approximation to it
* `λ`: estimate of the multiplier for the trust-radius constraint
* `Δ`: trust-radius value

**Returns**

* `p`: the step
* `f`: the value of the secular function
* `df`: derivative of `f` with respect to `λ`

"""
function secular(g::Array{Float64,1}, B::Array{Float64,2}, λ::Float64,
                 Δ::Float64; display_level::Int=0)
  @assert( norm(B - B.') < 100.*eps(), "B must be a symmetric matrix")
  @assert( size(B,1) == size(B,2) == size(g,1),
           "B must be a square matrix and/or the same size as g")
  @assert( Δ > 0 , "Δ must be a positive number" )
  n = size(g,1)
  diag_val = max(1.0, λ)*0.01*eps()
  semi_definite = true
  reg_iter = 1
  Bhat = zeros(B)
  local Fac::Base.Cholesky{Float64,Array{Float64,2}}
  while semi_definite
    try
      if reg_iter > 20
        throw(Base.ErrorException)
      end
      Bhat = B + eye(n)*(λ + diag_val)
      Fac = cholfact(Bhat)
      semi_definite = false
    catch exception
      if isa(exception, Base.PosDefException)
        diag_val *= 100.0
        if display_level > 1
          @printf("\tsecular: cholfact() failed, adding %g to diagonal\n",
                  diag_val)
        end
      else
        error("secular: regularization of Cholesky factorization failed")
      end
    end
    reg_iter += 1
  end
  
  # solve for the step; the step's length is used to define the secular function
  p = Fac\g
  scale!(p, -1.0) # to move g to rhs

  # compute the secular function
  norm_p = norm(p)
  f = 1.0/Δ - 1.0/norm_p

  # find the derivative of the secular function
  q = Fac[:L]\p
  df = norm(q)/norm_p
  df = -(df*df)/norm_p

  # return step, function, and derivative
  return p, f, df
end
