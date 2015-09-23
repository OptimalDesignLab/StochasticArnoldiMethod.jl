module Trust

export trust

  function trust(g,H,delta)
  
    println("hello")
    a = 2
    b = 22

    tol = 10.0^(-12)
    tol2 = 10.0^(-8)
    key = 0
    itbnd = 50
    lambda = 0
    n = length(g)
    # Matlab:
    #   coeff(1:n,1) = zeros(n,1)
    # TODO: fix
#     coeff[1:n,1] = zeros(n,1)

    # Matlab:
    #   H = full(H)
    # not needed in Julia?

    count = 0

    # slightly different than matlab version-
    #   matlab eig gives (V,D), where D is diag matrix
    #   julia eig gives (D,V), where D is a n-element one-dim array
    (eigval,V) = eig(H)
    #  so to convert to diagonal matrix:
    D = diagm(eigval)

    # matlab: [mineig, jmin] = min(eigval)
    (mineig, jmin) = findmin(eigval)

    alpha = -V'*g

    # matlab: sig = sign(alpha(jmin)) + (alpha(jmin)==0)
    # This makes an array of size alpha(jmin) that contains either a -1 or 1 depending
    #   on the sign of the array element of alpha(jmin). The == 0 term makes elements
    #   with value 0 have a sign of 1, otherwise when it is used in a multiplication 
    #   term below, it would zero out that value.
#     alpha_zeros = [int(alpha_el == 0) for alpha_el in alpha(jmin)]'
    alpha_zeros = int(alpha(jmin) .== 0)
    sig = sign(alpha(jmin)) + alpha_zeros

    # Positive definite case
    if mineig > 0
      coeff = alpha ./ eigval
      lambda = 0
      s = V*coeff
      posdef = 1
      nrms = norm(s)
      if nrms <= 1.2*delta
        key = 1
      else
        laminit = 0
      end
    else
      laminit = -mineig
      posdef = 0
    end   # end of 'if mineig > 0'

    if key == 0
      if seceqn(laminit,eigval,alpha,delta) > 0
        (b,c,count) = rfzero(laminit, itbnd, eigval, alpha, delta, tol)
        vval = abs(seceqn(b, eigval, alpha, delta))
        if vval <= tol2
          lambda = b
          key = 2
          lam = lambda*ones(n,1)
          w = eigval + lam
          #TODO verify these logical ANDs
          # Matlab:
          #   arg1 = (w == 0) & (alpha == 0)
          #   arg2 = (w == 0) & (alpha != 0)
          arg1 = int((w == 0) & (alpha == 0))
          arg2 = int((w == 0) & (alpha != 0))
          #TODO verify this conditional turds
          # Matlab:
          #   coeff(w != 0) = alpha(w != 0) ./ w(w != 0)
          #   coeff(arg1) = 0
          #   coeff(arg2) = Inf
          coeff[find(w)] = alpha[find(w)] ./ w[find(w)]
          coeff[arg1] = 0
          coeff[arg2] = Inf
          #TODO verify the isnan thing
          # Matlab:
          #   coeff(isnan(coeff)) = 0
          coeff[isnan(coeff)] = 0
          s = V*coeff
          #TODO verify norm functions equiv
          nrms = norm(s)
          if (nrms > 1.2*delta) || (nrms < 0.8*delta)
            key = 5
            lambda = -mineig
          end
        else
          lambda = -mineig
          key = 3
        end   # end of 'if vval <= tol2'
      else
        lambda = -mineig
        key = 4
      end   # end of 'if seceqn... > 0'
      lam = lambda*ones(n,1)
      if (key > 2)
        arg = abs(eigval + lam) < 10 * eps * max(abs(eigval),1)
        #TODO verify this arg business
        alpha(arg) = 0
      end   # end of 'if key > 2'
      w = eigval + lam
      #TODO another conditional and logical AND
      # Matlab:
      #   arg1 = (w == 0) & (alpha == 0) 
      #   arg2 = (w == 0) & (alpha != 0)
      arg1 = int((w == 0) & (alpha == 0))
      arg2 = int((w == 0) & (alpha != 0))
      # Matlab:
      #   coeff(w != 0) = alpha(w != 0) ./ w(w != 0)
      coeff[find(w)] = alpha[find(w)] ./ w[find(w)]
      # Matlab:
      #   coeff(arg1) = 0
      #   coeff(arg2) = Inf
      coeff[arg1] = 0
      coeff[arg2] = Inf
      #TODO isnan
      # Matlab:
      #   coeff(isnan(coeff)) = 0
      coeff[isnan(coeff)] = 0
      s = V*coeff
      nrms = norm(s)
      # line 95
      if (key > 2) && (nrms < 0.8*delta)
        beta = sqrt(delta^2 - nrms^2)
        s = s + beta*sig*V(:,jmin)
      end
      if (key > 2) && (nrms > 1.2*delta)
        [b,c,count] = rfzero(laminit, itbnd, eigval, alpha, delta, tol)
        lambda = b
        lam = lambda*(ones(n,1))
        w = eigval + lam
        #TODO another conditional and logical AND
        # Matlab:
        #   arg1 = (w == 0) & (alpha == 0) 
        #   arg2 = (w == 0) & (alpha != 0)
        arg1 = int((w == 0) & (alpha == 0))
        arg2 = int((w == 0) & (alpha != 0))
        # Matlab:
        #   coeff(w != 0) = alpha(w != 0) ./ w(w != 0)
        coeff[find(w)] = alpha[find(w)] ./ w[find(w)]
        # Matlab:
        #   coeff(arg1) = 0
        #   coeff(arg2) = Inf
        coeff[arg1] = 0
        coeff[arg2] = Inf
        # Matlab:
        #   coeff(isnan(coeff)) = 0
        coeff[isnan(coeff)] = 0
        s = V*coeff
        nrms = norm(s)
      end   # end of 'if (key > 2) && (nrms > 1.2*delta)'
    end   # end of 'if key == 0'


    val = g'*s + (0.5*s)' * (H*s)

    return s, val, posdef, count, lambda

  

  end   # end of function trust

  #--------------------------------------------------------------------
  # Secular equation
  # returns value of the secular equation at a set of m points lambda
  function seceqn(lambda,eigval,alpha,delta)
    m = length(lambda)
    n = length(eigval)
    unn = ones(n,1)
    unm = ones(m,1)
    M = eigval*unm' + unn*lambda
    MC = M
    MM = alpha*unm'

    # Matlab:
    #   M(M~=0) = MM(M~=0) ./M(M~=0)
    #   M(MC==0) = Inf
    # Julia: find(M) is sort of equivalent to M~=0; 
    #   it gives the indices of non-zero els
    M[find(M)] = MM[find(M)] ./ M[find(M)]
    M[find(MC)] = Inf

    M = M.*M
    value = sqrt(unm ./ (M'*unn))

    # Matlab:
    #   value(isnan(value)) = 0
    #   value = (1/delta)*unm - value
    value[find(isnan,value)] = 0
    value = (1/delta)*unm - value

    return value

  end   # end of function seceqn

  #--------------------------------------------------------------------
  # rfzero: find zero to the right
#   function rfzero(FunFcn,x,itbnd,eigval,alpha,delta,tol)
  function rfzero(x, itbnd, eigval, alpha, delta, tol)
  #DONE: fix this FunFcn business. Here it'll always be called with seceqn

    # Matlab: 
    #   if nargin < 7, tol = eps; end
    #   unneeded since rfzero will always be called with tol above

    itfun = 0

    if x != 0
      dx = abs(x)/2
    else
      dx = 1/2
    end

    a = x
    c = a

    # Matlab:
    #   fa = feval(FunFcn, a, eigval, alpha, delta)
    fa = seqeqn(a, eigval, alpha, delta)
    itfun = itfun + 1

    #TODO: why both of these? lines 181 & 182 in trust.m
    b = x + dx
    b = x + 1

    # Matlab:
    #   fb = feval(FunFcn, b, eigval, alpha, delta)
    fb = seceqn(b, eigval, alpha, delta)
    itfun = itfun + 1


    # Find change of sign
    while (fa > 0) == (fb > 0)

      dx = 2*dx
      if (fa > 0) != (fb > 0)
        break
      end
      b = x + dx
      fb = seqeqn(b, eigval, alpha, delta)
      itfun = itfun + 1
      if itfun > itbnd
        break
      end

    end   # end of first while loop

    fc = fb

    # Main loop, exit from middle of the loop
    while fb != 0
      # Ensure that b is the best result so far, a is the previous
      # value of b, and c is on the opposite of the zero from b
      if (fb > 0) == (fc > 0)
        c = a
        fc = fa
        d = b - a
        e = d
      end
      if abs(fc) < abs(fb)
        a = b
        b = c
        c = a
        fa = fb
        fb = fc
        fc = fa
      end
      
      # Convergence test and possible exit
      if itfun > itbnd
        break
      end
      m = 0.5*(c - b)
      toler = 2.0*tol*max(abs(b),1.0)
      if (abs(m) <= toler) || (fb == 0.0)
        break
      end
  
      # Choose bisection or interpolation
      if (abs(e) < toler) || (abs(fa) <= abs(fb))
        # Bisection
        d = m
        e = m
      else
        # Interpolation
        s = fb/fa
        if (a == c)
          # Linear interpolation
          p = 2.0*m*s
          q = 1.0 - s
        else
          # Inverse quadratic interpolation
          q = fa/fc
          r = fb/fc
          p = s*(2.0*m*q*(q - r) - (b - a)*(r - 1.0))
          q = (q - 1.0)*(r - 1.0)*(s - 1.0)
        end
        if p > 0
          q = -q
        else
          p = -p
        end
        # Is interpolated point acceptable
        if (2.0*p < 3.0*m*q - abs(toler*q)) && (p < abs(0.5*e*q))
          e = d
          d = p/q
        else
          d = m
          e = m
        end
      end # Interpolation
  
      # Next point
      a = b
      fa = fb
      if abs(d) > toler
        b = b + d
      elseif b > c 
        b = b - toler
      else 
        b = b + toler
      end
      seqeqn(b, eigval, alpha, delta)
      itfun = itfun + 1
    end   # end of while loop

  end   # end of function rfzero

end
