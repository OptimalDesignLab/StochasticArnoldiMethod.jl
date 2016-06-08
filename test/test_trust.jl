facts("Testing StochasticArnoldiMethod Module (trust.jl file)...") do

  context("Testing secular (nonsymmetric matrix)") do
    # test for failure in the case of a nonsymmetric matrix
    n = 3
    B = rand(n,n)
    g = rand(n)
    λ = 1.0
    Δ = 1.0
    @fact_throws secular(g, B, λ, Δ)
  end
    
  context("Testing secular (highly negative definite matrix)") do
    # test for failure in the case of a negative definite matrix
    n = 3
    Q, R = qr(rand(n,n))
    B = Q*(diagm([-1e100; -1e200; -1e300])*Q.')
    g = rand(n)
    λ = 1.0
    Δ = 1.0
    @fact_throws secular(g, B, λ, Δ)
  end

  context("Testing secular (positive definite matrix)") do
    # test for failure in the case of a nonsymmetric matrix
    n = 3
    Q, R = qr(rand(n,n))
    B = Q*diagm([1., 10., 100.])*Q.'
    g = rand(n)
    λ = 1.0
    Δ = 1.0
    p, f, df = secular(g, B, λ, Δ)
    A = B + eye(n)*λ
    p_true = -A\g
    @fact p --> roughly(p_true, atol=1e-12)
    @fact f --> roughly(1.0/Δ - 1./norm(p_true), atol=1e-12)
    @fact df --> roughly( -dot(p_true, A\p_true)/(norm(p_true)^(3)), atol=1e-12)
  end
  
  context("Testing trust (trust-region constraint inactive)") do
    B = [3.931544008059447 -4.622828930484834 1.571893108754884;
         -4.622828930484834 5.438436601890520 -1.853920290644159;
         1.571893108754884 -1.853920290644159 0.640029390050034]
    Δ = 1e+6
    g = [-0.964888535199277; -0.157613081677548; -0.970592781760616]
    p, pred, active = trust(g, B, Δ)
    @fact p --> roughly([70306.51597578949; 71705.07007628426; 
                         35032.96463402245], rtol=sqrt(eps()))
    @fact pred --> roughly(56571.17543987902, rtol=sqrt(eps()))
    @fact active --> false
  end

  context("Testing trust (trust-region constraint active)") do
    B = [3.931544008059447 -4.622828930484834 1.571893108754884;
         -4.622828930484834 5.438436601890520 -1.853920290644159;
         1.571893108754884 -1.853920290644159 0.640029390050034]
    Δ = 10000.0
    g = [-0.964888535199277; -0.157613081677548; -0.970592781760616]
    p, pred, active = trust(g, B, Δ, display_level=0)
    @fact p --> roughly([6592.643411070546; 6740.041501057747; 
                         3333.0006627887597], rtol=sqrt(eps()))
    @fact pred --> roughly(10147.173335581543, rtol=sqrt(eps()))
    @fact active --> true
  end

  context("Testing trust (indefinite Hessian)") do
    B = [3.931535263699851 -4.622837846534464 1.571888758188687;
         -4.622837846534464 5.438427510779841 -1.853924726631001;
         1.571888758188687 -1.853924726631001 0.640027225520312]
    g = -(1e-5)*ones(3)
    Δ = 10000.0
    p, pred, active = trust(g, B, Δ, display_level=0)
    @fact p --> roughly([6612.245665627993; 6742.073145692998;
                         3289.7797283358564], rtol=sqrt(eps()))
    @fact pred --> roughly(500.1664095358195, rtol=sqrt(eps()))
    @fact active --> true
  end

end