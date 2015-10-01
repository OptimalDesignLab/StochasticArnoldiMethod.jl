function ApproxArnoldi(fun, x0, fun0, grad0, delta, rank, m)
# estimate the largest rank eigenvalues/vectors of the Hessian of fun
# inputs:
#   fun - function handle of function whose Hessian we want to model
#   x0 - base point about which differences are formed
#   fun0 - initial function value
#   grad0 - initial gradient; used as seed for Arnoldi's method
#   delta - finite-difference step size
#   rank - rank of the Hessian
#   m - number of Arnoldi iterations (number of samples-1)
# outputs:
#   xdata, fdata, gdata - data locations, function data, gradient data
#   E - approximate eigenvectors
#   L - approximate eigenvalues
#   new_rank - if Arnoldi breaks down, this gives the new rank
#   gred - the reduced gradient approximated using function differences
#--------------------------------------------------------------------------

n = size(grad0, 1)

# setup data arrays
xdata = zeros(n, m+1)
fdata = zeros(1, m+1)
gdata = zeros(n, m+1)
fdata[1,1] = fun0
gdata[:,1] = grad0

V = zeros(n, m+1)
H = zeros(m+1, m)
V[:,1] = grad0 ./ norm(grad0)
new_rank = rank

for i = 1:m

  # find new sample point and data
  xdata[:,i+1] = V[:,i] * delta
  x = x0 + xdata[:, i+1]
  (fdata[1,i+1], gdata[:,i+1]) = fun(x)

  # find new basis vector
  V[:,i+1] = (gdata[:,i+1] - gdata[:,1]) ./ delta
  nrm = norm(V[:,i+1])^2
  thr = nrm*0.98
  #TODO: find out why this is

  for j = 1:i
    H[j,i] = V[:,i+1].' * V[:,j]
    V[:,i+1] = V[:,i+1] - H[j,i]*V[:,j]

    if (H[j,i]*H[j,i] > thr)
      prod = V[:,i+1].' * V[:,j]
      H[j,i] = H[j,i] + prod
      V[:,i+1] = V[:,i+1] - prod*V[:,j]
    end

    nrm = nrm - H[j,i]*H[j,i]
    nrm = max(nrm, 0.0)
    thr = nrm * 0.98
  end

  H[i+1,i] = norm(V[:,i+q])
  V[:,i+1] = V[:,i+1] / H[i+1,i]
  # at line 56 in ApproxArnoldi.m
  if (H[i+1,i] < 1e-14)
    # break down
    new_rank = i
    println("Arnoldi-sampling breakdown: iteration ",i)
    break
  end
end

# Matlab:
#   [Ered, Lred] = eig(0.5*(H(1:i,1:i) + H(1:i,1:i).'));
#   Lred = diag(Lred);
(Lred, Ered) = eig(0.5*(H[1:i,1:i] + H[1:i,1:i].'))

# Matlab:
#   [~, indx] = sort(abs(Lred),'descend');
#   Lred = Lred(indx);
# instead, use sortperm to give indices
indx = sortperm(abs(Lred))
Lred = Lred[indx]
L = Lred[1:new_rank]
Ered = Ered[:,indx]
E = zeros(n, new_rank)
for k = 1:new_rank
  E[:,k] = V[:,1:i]*Ered[1:i,k]
end
tmp = (fdata[1,2:i+1].' - ones(i,1)*fdata[1,1]) ./ delta
gred = Ered[1:i,1:new_rank].'*tmp

end





