@doc """
### StochasticArnoldiMethod.arnoldiSample

Generates a sample of points, function values, and gradients using arnoldi
sampling.  Also returns a quadratic model based on the sampled gradients.

**Inputs**

* `func`: functor that returns function and gradient information
* `alpha`: pertubation size used to generate samples
* `num_sample`: number of samples (number of Arnoldi iterations + 1)

**In/Outs**

* `xdata`: array of sampled locations; on entry, xdata[:,1] is initial sample
* `fdata`: array of sampled function values; fdata[:,1] is the initial function
* `gdata`: array of sampled gradient values; gdata[:,1] is the initial gradient

**Outputs**

* `eigenvals`: eigenvalues of the quadratic model
* `eigenvecs`: eigenvectors of the quadratic model
* `grad_red`: the reduced gradient approximated using directional derivatives

**Returns**

* the size of the subspace; this may not equal num_sample-1

"""->
function arnoldiSample(func::Function, xdata, fdata, gdata, alpha, num_sample,
                       eigenvals, eigenvecs, grad_red)
  n = size(xdata,1)
  m = num_sample-1
  @assert(size(gdata,1) == size(eigenvecs,1) == n)
  @assert(size(xdata,2) == size(fdata,2) == size(gdata,2) == num_sample)
  @assert(alpha > 0)

  # initialize the basis-vector array and Hessenberg matrix
  V = zeros(n, m+1)
  H = zeros(m+1,m) 
  V[:,1] = gdata[:,1]./norm(gdata[:,1])
  
  lin_depend = false
  i = 0
  for i = 1:m

    # find new sample point and data
    xdata[:,i+1] = xdata[:,1] + V[:,i] * alpha
    func(view(xdata,:,i+1), view(fdata,:,i+1), view(gdata,:,i+1))

    # find the new basis vector and orthogonalize it against the old ones
    V[:,i+1] = (gdata[:,i+1] - gdata[:,1])./alpha
    lin_depend = modGramSchmidt(i, H, V)
    if lin_depend
      # new basis vector is linealy dependent, so terminate early
      break
    end
  end
  if lin_depend
    i -= 1
  end

  # symmetrize the Hessenberg matrix, and find its eigendecomposition
  Hsym = Symmetric(0.5.*(H[1:i,1:i] + H[1:i,1:i]))
  fill!(eigenvals, 0.0)
  eigenvals[1:i], eigvecs_red = eig(Hsym)

  # generate the full-space eigenvector approximations
  fill!(eigenvecs, 0.0)
  for k = 1:i
    eigenvecs[:,k] = V[:,1:i]*eigvecs_red[1:i,k]
  end

  # generate the directional-derivative approximation to the reduced gradient
  tmp = (fdata[1,2:i+1].' - ones(i,1)*fdata[1,1])./alpha
  gred = eigvecs_red[1:i,1:i].'*tmp

  return i
end

@doc """
### StochasticArnoldiMethod.modGramSchmidt

Performs modified Gram-Schmidt on the i+1 column of w, with respect to the
previous 1:i columns.  Reorthogonalization is used if a threshold is met.  Based
on Kesheng John Wu's mgsro subroutine in Saad's SPARSKIT.

**Inputs**

* `i`: the `i+1` column is orthogonalized

**In/Outs**

* `Hsbg`: the upper Hessenberg matrix generated during Arnoldi's method
* `w`: contains the vectors being orthonalized

**Returns**

* `True` if the vector `w[:,i+1]` is linearly dependent, `False` otherwise

"""->
function modGramSchmidt(i, Hsbg, w)
  @assert(size(w,2) >= i+1)
  @assert(size(Hsbg,1) >= i+1)
  @assert(size(Hsbg,2) >= i)

  err_msg = "modGramSchmidt failed: "
  const reorth = 0.98

  # get the norm of the vector being orthogonalized, and find the threshold for
  # re-orthogonalization
  nrm = dot(w[:,i+1],w[:,i+1])
  thr = nrm*reorth
  if abs(nrm) <= 10.*eps(Hsbg[1,1]) 
    # the norm of w[i+1] is effectively zero; it is linearly dependent
    return true
  elseif nrm < -eps(Hsbg[1,1])
    # the norm of w[i+1] < 0.0
    err_msg *= "InnerProd(w[i+1], w[i+1]) = "*string(nrm)*" < 0.0"
    error(err_msg)
    return false
  elseif nrm != nrm
    # this is intended to catch if nrm = NaN, but some optimizations may mess it
    # up (according to posts on stackoverflow.com)
    err_msg *= "w[i+1] = NaN"
    error(err_msg)
    return false
  end

  if i < 1
    # just normalize and exit
    w[:,i+1] /= sqrt(nrm)
    return false
  end
  
  # begin main Gram-Schmidt loop
  for k = 1:i
    prod = dot(w[:,i+1], w[:,k])
    Hsbg[k,i] = prod
    w[:,i+1] -= prod*w[:,k]
    # check if reorthogonalization is necessary
    if prod*prod > thr
      prod = dot(w[:,i+1], w[:,k])
      Hsbg[k,i] += prod
      w[:,i+1] -= prod*w[:,k]
    end
    # update the norm and check its size
    nrm -= Hsbg[k,i]*Hsbg[k,i]
    nrm < 0.0 ? nrm = 0.0 : nothing
    thr = nrm*reorth
  end
  
  # test the resulting vector
  nrm = norm(w[:,i+1])
  Hsbg[i+1,i] = nrm
  if nrm <= 10.0*eps(Hsbg[1,1])
    return true
  else
    # scale the vector
    w[:,i+1] /= nrm
    return false
  end
end

