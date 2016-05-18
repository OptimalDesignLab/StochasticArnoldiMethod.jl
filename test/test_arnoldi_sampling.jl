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
  

  #context("Testing arnoldiSampling") do


  #end

end