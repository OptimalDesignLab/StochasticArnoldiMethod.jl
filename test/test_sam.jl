facts("Testing StochasticArnoldiMethod Module (sam.jl file)...") do

  context("Testing SAM (positive-definite quadratic, no error)") do
    # This tests that SAM reverts to Newton-CG
    n = 100
    V, R = qr(rand(n,n))
    E = sort(rand(n))
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
    x0 = ones(n,1)./sqrt(n)
    tol = 1e-8
    options = default_options
    options["display_level"] = 0 #2
    options["alpha"] = sqrt(eps()) # FD method
    options["max_iter"] = 20
    options["num_sample"] = 20
    x, f, hist = SAM(quad, x0, tol, options)
    #println("norm(x) = ",norm(x))
    #println("f = ",f)
    @fact norm(x) --> roughly(0.0, atol=1e-6)
    @fact f --> roughly(0.0, atol=1e-12)
  end

  context("Testing SAM (positive-definite quadratic with error)") do
    n = 100
    V, R = qr(rand(n,n))
    E = sort(rand(n))
    func_std = 0.0
    grad_std = 0.0
    function quad(x::AbstractArray, f::AbstractArray, g::AbstractArray;
                  exact::Bool=false)
      Vx = V.'*x
      fill!(g, 0.0)
      f[:] = 0.0
      for i = 1:n
        for j = 1:n
          g[i] += V[i,j]*E[j]*Vx[j]
        end
        f[:] += 0.5*g[i]*x[i]
      end
      if !exact
        f[:] += func_std*randn((1))
        g[:,1] += grad_std*randn((n))
      end
    end
    x0 = ones(n,1)./sqrt(n)
    f = zeros(1,1)
    g = zeros(n,1)
    quad(x0, f, g)
    func_std = 0.01*f
    grad_std = 0.01*norm(g)
    println("func_std = ",func_std)
    println("grad_std = ",grad_std)
    tol = 1e-6
    options = default_options
    options["display_level"] = 1
    options["alpha"] = 1.0
    options["max_iter"] = 20
    options["num_sample"] = 20
    #options["grad_method"] = "dirderiv"
    x, f, hist = SAM(quad, x0, tol, options)
    println("norm(x) = ",norm(x))
    println("f = ",f)
    #@fact norm(x) --> roughly(0.0, atol=tol)
    #@fact f --> roughly(0.0, atol=1e-14)
  end

  

end