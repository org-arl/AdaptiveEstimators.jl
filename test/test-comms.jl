using AdaptiveEstimators
using StableRNGs
using Statistics
using Test

function channel_estimation(T, ch, algo; M=16, N=1024, σ=1f-2, nsteps=N, saveat=0)
  rng = StableRNG(1)
  x = randn(rng, T, N)
  h = zeros(T, M)
  y = mapreduce(+, ch) do (i, a)
    h[i+1] = a
    conj(a) * vcat(zeros(T, i), x[1:end-i])
  end
  y += σ * randn(rng, T, N)
  r = fit!(LinearModel(T, M), algo, x, y, nsteps; saveat, rng)
  @test length(r.p) == M
  @test length(r.loss) == nsteps
  @test length(r.y) == nsteps
  @test size(r.phist, 2) == (saveat == 0 ? 0 : nsteps ÷ saveat)
  @test eltype(r.p) == T
  @test eltype(r.y) == T
  @test r.loss[end] < 3σ^2
  @test mean(abs2, r.y[N÷2:end] - y[N÷2:nsteps]) < 3σ^2
  @test r.p ≈ h atol=3σ
end

function channel_equalization(ch, eq, algo; M=64, nsym=8192, ntrain=512, σ=1f-2, PSK=4)
  Q = cispi.(2 .* (0:PSK-1) ./ PSK)
  rng = StableRNG(1)
  x = rand(rng, Q, nsym)
  h = zeros(ComplexF64, M)
  y = mapreduce(+, ch) do (i, a)
    h[i+1] = a
    conj(a) * vcat(zeros(ComplexF64, i), x[1:end-i])
  end
  y += σ * randn(rng, ComplexF64, nsym)
  decision = nearest(Q)
  r = fit!(eq(ComplexF64, M), algo, y, x[1:ntrain], length(x), decision; rng)
  ser = count(decision.(r.y[ntrain+1:end]) .!= x[ntrain+1:end]) / (nsym - ntrain)
  @test length(r.loss) == nsym
  @test length(r.y) == nsym
  @test ser < 0.02
end
