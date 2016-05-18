facts("Testing StochasticArnoldiMethod Module (arnoldi_sampling.jl file)...") do

  context("Testing modGramSchmidt (full rank case)") do
    # generate a random set of vectors and use modGramSchmidt to orthogonalize
    # them
    n = 10
    m = 4
    V = rand(n,m)
    H = zeros(m,m-1)
    for i = 0:m-1
      StochasticArnoldiMethod.modGramSchmidt(i, H, V)
      @fact norm(V[:,i+1]) --> roughly(1.0, atol=1e-14)
    end
    # check that vectors are orthogonal
    for i = 1:m
      for j = i+1:m
        @fact dot(V[:,i],V[:,j]) --> roughly(0.0, atol=1e-14)
      end
    end
  end

  context("Testing modGramSchmidt (rank deficient case)") do
    # generate a random set of vectors, make one of them a linear combination of
    # the others, and use modGramSchmidt to orthogonalize them
    n = 10
    m = 4
    V = rand(n,m)
    V[:,m] = V[:,1:m-1]*rand(m-1,1)
    H = zeros(m,m-1)
    for i = 0:m-2
      StochasticArnoldiMethod.modGramSchmidt(i, H, V)
      @fact norm(V[:,i+1]) --> roughly(1.0, atol=1e-14)
    end
    # calling now should produce lin_depend flag
    lin_depend = StochasticArnoldiMethod.modGramSchmidt(m-1, H, V)
    @fact lin_depend --> true
  end

  context("Testing arnoldiSample (positive definite quadratic)") do
    # use a synthetic quadratic function, and check that arnoldiSample recovers
    # its eigenvalues and eigenvectors
    n = 10
    V, R = qr(rand(n,n))
    E = sort(rand(10))
    function quad(x::AbstractArray, f::AbstractArray, g::AbstractArray)
      Vx = V.'*x
      fill!(g, 0.0)
      f[:] = 0.0
      for i = 1:n
        for j = 1:n
          g[i] += V[i,j]*E[j]*Vx[j]
        end
        f[:] += 0.5*g[i]*x[i]
      end
    end

    # generate data at initial point (1,1,1,...,1)^T
    xdata = zeros(n,n+1); fdata = zeros(1,n+1); gdata = zeros(n,n+1)
    xdata[:,1] = ones(n)
    quad(view(xdata,:,1), view(fdata,:,1), view(gdata,:,1))

    # generate sample
    alpha = 1.0
    num_sample = n+1
    eigenvals = zeros(n)
    eigenvecs = zeros(n,n)
    grad_red = zeros(n)
    arnoldiSample(quad, xdata, fdata, gdata, alpha, num_sample,
                  eigenvals, eigenvecs, grad_red)
    
    # check that eigenvalues and eigenvectors agree
    for i = 1:n
      @fact eigenvals[i] --> roughly(E[i], atol=1e-7)
      @fact abs(dot(eigenvecs[:,i],V[:,i])) --> roughly(1.0, atol=1e-7)
    end
  end

  context("Testing arnoldiSample (semi-definite quadratic)") do
    # use a synthetic quadratic function, and check that arnoldiSample recovers
    # its eigenvalues and eigenvectors
    n = 10
    V, R = qr(rand(n,n))
    E = sort(rand(10))
    E[1] = 0.0
    function quad(x::AbstractArray, f::AbstractArray, g::AbstractArray)
      Vx = V.'*x
      fill!(g, 0.0)
      f[:] = 0.0
      for i = 1:n
        for j = 1:n
          g[i] += V[i,j]*E[j]*Vx[j]
        end
        f[:] += 0.5*g[i]*x[i]
      end
    end

    # generate data at initial point (1,1,1,...,1)^T
    xdata = zeros(n,n+1); fdata = zeros(1,n+1); gdata = zeros(n,n+1)
    xdata[:,1] = ones(n)
    quad(view(xdata,:,1), view(fdata,:,1), view(gdata,:,1))

    # generate sample
    alpha = 1.0
    num_sample = n+1
    eigenvals = zeros(n)
    eigenvecs = zeros(n,n)
    grad_red = zeros(n)
    dim = arnoldiSample(quad, xdata, fdata, gdata, alpha, num_sample,
                        eigenvals, eigenvecs, grad_red)
    @fact dim --> n-1
    
    # check that eigenvalues and eigenvectors agree
    for i = 1:dim
      @fact eigenvals[i] --> roughly(E[i+1], atol=1e-7)
      @fact abs(dot(eigenvecs[:,i],V[:,i+1])) --> 
      roughly(1.0, atol=1e-7)
    end
  end

  context("Testing arnoldiSample (reduced gradient)") do
    # use a nonlinear function and a small perturbation to test the reduced
    # gradient produced by arnoldiSample
    n = 10
    function nonlinear(x::AbstractArray, f::AbstractArray, g::AbstractArray)
      xs = x./[1:n;]
      f[1] = exp(dot(xs,xs))
      for i = 1:n
        g[i] = 2.0*x[i]*f[1]./(i*i)
      end
    end
    
    # generate data at initial point (1,1,1,...,1)^T
    xdata = zeros(n,n+1); fdata = zeros(1,n+1); gdata = zeros(n,n+1)
    xdata[:,1] = ones(n)
    nonlinear(view(xdata,:,1), view(fdata,:,1), view(gdata,:,1))

    # generate sample
    alpha = sqrt(eps(1.0))
    num_sample = n+1
    eigenvals = zeros(n)
    eigenvecs = zeros(n,n)
    grad_red = zeros(n)
    dim = arnoldiSample(nonlinear, xdata, fdata, gdata, alpha, num_sample,
                        eigenvals, eigenvecs, grad_red)

    # check reduced gradient;
    # rotate back out of eigenvector coordinates
    g = eigenvecs*grad_red
    @fact g --> roughly(gdata[:,1], atol=1e-6)
  end

end