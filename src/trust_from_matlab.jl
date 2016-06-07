@doc """
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

"""->
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
      return p, pred
    end
    if display_level > 0
      println("\ttrust: norm(p) = ",norm(p)," > ",Δ)
    end
  end
  # if we get here, either the Hessian is semi-definite or ||p|| > radius

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
        @printf("\trust: Newton converged with lambda = %g\n", λ)
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
  return p, pred
end

@doc """
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

"""->
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

# module Trust

# export trust

# function trust(g,H,delta)

#   println("hello")
#   a = 2
#   b = 22

#   tol = 10.0^(-12)
#   tol2 = 10.0^(-8)
#   key = 0
#   itbnd = 50
#   lambda = 0
#   n = length(g)
#   # Matlab:
#   #   coeff(1:n,1) = zeros(n,1)
#   # TODO: fix
#   coeff = zeros(n,1)
#   coeff[1:n,1] = zeros(n,1)

#   # Matlab:
#   #   H = full(H)
#   # not needed in Julia?

#   count = 0

#   # slightly different than matlab version-
#   #   matlab eig gives (V,D), where D is diag matrix
#   #   julia eig gives (D,V), where D is a n-element one-dim array
#   (eigval,V) = eig(H)
#   #  so to convert to diagonal matrix:
#   D = diagm(eigval)

#   # matlab: [mineig, jmin] = min(eigval)
#   (mineig, jmin) = findmin(eigval)

#   alpha = -V'*g

#   # matlab: sig = sign(alpha(jmin)) + (alpha(jmin)==0)
#   # This makes an array of size alpha(jmin) that contains either a -1 or 1 depending
#   #   on the sign of the array element of alpha(jmin). The == 0 term makes elements
#   #   with value 0 have a sign of 1, otherwise when it is used in a multiplication 
#   #   term below, it would zero out that value.
#   # alpha_zeros = [int(alpha_el == 0) for alpha_el in alpha(jmin)]'
#   alpha_zeros = int(alpha[jmin] .== 0)
#   sig = sign(alpha[jmin]) + alpha_zeros

#   # Positive definite case
#   if mineig > 0
#     coeff = alpha ./ eigval
#     lambda = 0
#     s = V*coeff
#     posdef = 1
#     nrms = norm(s)
#     if nrms <= 1.2*delta
#       key = 1
#     else
#       laminit = 0
#     end
#   else
#     laminit = -mineig
#     posdef = 0
#   end   # end of 'if mineig > 0'

#   if key == 0
#     seceqn_result = seceqn(laminit,eigval,alpha,delta) 
#     #TODO bug here
# #     println(typeof(seceqn_result))
# #     println(seceqn_result)
#     if seceqn_result > 0.0
#       (b,c,count) = rfzero(laminit, itbnd, eigval, alpha, delta, tol)
#       vval = abs(seceqn(b, eigval, alpha, delta))
#       if vval <= tol2
#         lambda = b
#         key = 2
#         lam = lambda*ones(n,1)
#         w = eigval + lam
#         #TODO verify these logical ANDs
#         # Matlab:
#         #   arg1 = (w == 0) & (alpha == 0)
#         #   arg2 = (w == 0) & (alpha != 0)
#         arg1 = int((w == 0) & (alpha == 0))
#         arg2 = int((w == 0) & (alpha != 0))
#         #TODO verify this conditional turds
#         # Matlab:
#         #   coeff(w != 0) = alpha(w != 0) ./ w(w != 0)
#         #   coeff(arg1) = 0
#         #   coeff(arg2) = Inf
#         coeff[find(w)] = alpha[find(w)] ./ w[find(w)]
#         coeff[arg1] = 0
#         coeff[arg2] = Inf
#         #TODO verify the isnan thing
#         # Matlab:
#         #   coeff(isnan(coeff)) = 0
#         coeff[isnan(coeff)] = 0
#         s = V*coeff
#         #TODO verify norm functions equiv
#         nrms = norm(s)
#         if (nrms > 1.2*delta) || (nrms < 0.8*delta)
#           key = 5
#           lambda = -mineig
#         end
#       else
#         lambda = -mineig
#         key = 3
#       end   # end of 'if vval <= tol2'
#     else
#       lambda = -mineig
#       key = 4
#     end   # end of 'if seceqn... > 0'
#     lam = lambda*ones(n,1)
#     if (key > 2)
#       # Matlab:
#       #   arg = abs(eigval + lam) < 10 * eps * max(abs(eigval),1)
#       #TODO FIX ALL THIS
# #       println(typeof(abs(eigval + lam)))
# #       println(abs(eigval + lam))
# #       println(typeof(10 * eps(Float64) * max(abs(eigval),1)))
# #       println(10 * eps(Float64) * max(abs(eigval),1))
# #       arg = float(abs(eigval + lam)) < float(10 * eps(Float64) * max(abs(eigval),1))
#       #TODO verify this arg business
# #       alpha(arg) = 0
#     end   # end of 'if key > 2'
#     w = eigval + lam
#     #TODO another conditional and logical AND
#     # Matlab:
#     #   arg1 = (w == 0) & (alpha == 0) 
#     #   arg2 = (w == 0) & (alpha != 0)
#     arg1 = int((w == 0) & (alpha == 0))
#     arg2 = int((w == 0) & (alpha != 0))
#     # Matlab:
#     #   coeff(w != 0) = alpha(w != 0) ./ w(w != 0)
#     coeff[find(w)] = alpha[find(w)] ./ w[find(w)]
#     # Matlab:
#     #   coeff(arg1) = 0
#     #   coeff(arg2) = Inf
# #     coeff[arg1] = 0
# #     coeff[arg2] = Inf
#     coeff[arg1+1] = 0
#     coeff[arg2+1] = Inf
#     #TODO isnan
#     # Matlab:
#     #   coeff(isnan(coeff)) = 0
#     coeff[isnan(coeff)] = 0
#     s = V*coeff
#     nrms = norm(s)
#     # line 95
#     if (key > 2) && (nrms < 0.8*delta)
#       beta = sqrt(delta^2 - nrms^2)
#       s = s + beta*sig*V(:,jmin)
#     end
#     if (key > 2) && (nrms > 1.2*delta)
#       (b,c,count) = rfzero(laminit, itbnd, eigval, alpha, delta, tol)
#       lambda = b
#       lam = lambda*(ones(n,1))
#       w = eigval + lam
#       #TODO another conditional and logical AND
#       # Matlab:
#       #   arg1 = (w == 0) & (alpha == 0) 
#       #   arg2 = (w == 0) & (alpha != 0)
#       arg1 = int((w == 0) & (alpha == 0))
#       arg2 = int((w == 0) & (alpha != 0))
#       # Matlab:
#       #   coeff(w != 0) = alpha(w != 0) ./ w(w != 0)
#       coeff[find(w)] = alpha[find(w)] ./ w[find(w)]
#       # Matlab:
#       #   coeff(arg1) = 0
#       #   coeff(arg2) = Inf
#       coeff[arg1] = 0
#       coeff[arg2] = Inf
#       # Matlab:
#       #   coeff(isnan(coeff)) = 0
#       coeff[isnan(coeff)] = 0
#       s = V*coeff
#       nrms = norm(s)
#     end   # end of 'if (key > 2) && (nrms > 1.2*delta)'
#   end   # end of 'if key == 0'


#   val = g'*s + (0.5*s)' * (H*s)

#   return s, val, posdef, count, lambda



# end   # end of function trust
# #--------------------------------------------------------------------
# # Secular equation
# # returns value of the secular equation at a set of m points lambda
# function seceqn(lambda,eigval,alpha,delta)
#   m = length(lambda)
#   n = length(eigval)
#   unn = ones(n,1)
#   unm = ones(m,1)
#   M = eigval*unm' + unn*lambda
#   MC = M
#   MM = alpha*unm'

#   # Matlab:
#   #   M(M~=0) = MM(M~=0) ./M(M~=0)
#   #   M(MC==0) = Inf
#   # Julia: find(M) is sort of equivalent to M~=0; 
#   #   it gives the indices of non-zero els
#   M[find(M)] = MM[find(M)] ./ M[find(M)]
#   M[find(MC)] = Inf

#   M = M.*M
#   value = sqrt(unm ./ (M'*unn))

#   # Matlab:
#   #   value(isnan(value)) = 0
#   #   value = (1/delta)*unm - value
#   value[find(isnan,value)] = 0
#   value = (1/delta)*unm - value

#   # Added because value becomes Array{Float64,2}
#   value = int(value[1])

#   return value

# end   # end of function seceqn


# #--------------------------------------------------------------------
# # rfzero: find zero to the right
# function rfzero(x, itbnd, eigval, alpha, delta, tol)

#   # Matlab: 
#   #   if nargin < 7, tol = eps; end
#   #   unneeded since rfzero will always be called with tol above

#   itfun = 0

#   if x != 0
#     dx = abs(x)/2
#   else
#     dx = 1/2
#   end

#   a = x
#   c = a

#   # Matlab:
#   #   fa = feval(FunFcn, a, eigval, alpha, delta)
#   fa = seqeqn(a, eigval, alpha, delta)
#   itfun = itfun + 1

#   #TODO: why both of these? lines 181 & 182 in trust.m
#   b = x + dx
#   b = x + 1

#   # Matlab:
#   #   fb = feval(FunFcn, b, eigval, alpha, delta)
#   fb = seceqn(b, eigval, alpha, delta)
#   itfun = itfun + 1


#   # Find change of sign
#   while (fa > 0) == (fb > 0)

#     dx = 2*dx
#     if (fa > 0) != (fb > 0)
#       break
#     end
#     b = x + dx
#     fb = seqeqn(b, eigval, alpha, delta)
#     itfun = itfun + 1
#     if itfun > itbnd
#       break
#     end

#   end   # end of first while loop

#   fc = fb

#   # Main loop, exit from middle of the loop
#   while fb != 0
#     # Ensure that b is the best result so far, a is the previous
#     # value of b, and c is on the opposite of the zero from b
#     if (fb > 0) == (fc > 0)
#       c = a
#       fc = fa
#       d = b - a
#       e = d
#     end
#     if abs(fc) < abs(fb)
#       a = b
#       b = c
#       c = a
#       fa = fb
#       fb = fc
#       fc = fa
#     end
    
#     # Convergence test and possible exit
#     if itfun > itbnd
#       break
#     end
#     m = 0.5*(c - b)
#     toler = 2.0*tol*max(abs(b),1.0)
#     if (abs(m) <= toler) || (fb == 0.0)
#       break
#     end

#     # Choose bisection or interpolation
#     if (abs(e) < toler) || (abs(fa) <= abs(fb))
#       # Bisection
#       d = m
#       e = m
#     else
#       # Interpolation
#       s = fb/fa
#       if (a == c)
#         # Linear interpolation
#         p = 2.0*m*s
#         q = 1.0 - s
#       else
#         # Inverse quadratic interpolation
#         q = fa/fc
#         r = fb/fc
#         p = s*(2.0*m*q*(q - r) - (b - a)*(r - 1.0))
#         q = (q - 1.0)*(r - 1.0)*(s - 1.0)
#       end
#       if p > 0
#         q = -q
#       else
#         p = -p
#       end
#       # Is interpolated point acceptable
#       if (2.0*p < 3.0*m*q - abs(toler*q)) && (p < abs(0.5*e*q))
#         e = d
#         d = p/q
#       else
#         d = m
#         e = m
#       end
#     end # Interpolation

#     # Next point
#     a = b
#     fa = fb
#     if abs(d) > toler
#       b = b + d
#     elseif b > c 
#       b = b - toler
#     else 
#       b = b + toler
#     end
#     #TODO
#     seqeqn(b, eigval, alpha, delta)
#     itfun = itfun + 1
#   end   # end of while loop

#   return b, c, itfun

# end   # end of function rfzero

# end   # end of module definition
