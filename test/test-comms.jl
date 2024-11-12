using AdaptiveEstimators
using StableRNGs
using Statistics
using Test

function channel_estimation(T, ch, algo; M=16, N=1024, σ=1f-2, ts=1:N, saveat=0)
  rng = StableRNG(1)
  x = randn(rng, T, N)
  h = zeros(T, M)
  y = mapreduce(+, ch) do (i, a)
    h[i+1] = a
    conj(a) * vcat(zeros(T, i), x[1:end-i])
  end
  y += σ * randn(rng, T, N)
  r = fit!(LinearModel(T, M), algo, (x, y), ts; saveat, rng)
  @test length(r.ps) == M
  @test length(r.loss) == length(ts)
  @test length(r.out) == length(ts)
  @test length(r.ts) == (saveat == 0 ? 0 : length(ts) ÷ saveat)
  @test size(r.ps_history) == (M, length(r.ts))
  @test eltype(r.ps) == T
  @test eltype(r.out) == T
  @test r.loss[end] < 3σ^2
  @test mean(abs2, r.out[N÷2:end] - y[N÷2:ts[end]]) < 3σ^2
  @test r.ps ≈ h atol=3σ
end

function channel_equalization(ch, algo; M=64, nsym=8192, ntrain=512, σ=1f-2, PSK=4)
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
  r = fit!(LinearModel(ComplexF64, M, decision), algo, (y, x[1:ntrain]), 1:length(x); rng)
  ser = count(decision.(r.out[ntrain+1:end]) != x[ntrain+1:end]) / (nsym - ntrain)
  @test length(r.ps) == M
  @test length(r.loss) == nsym
  @test length(r.out) == nsym
  @test ser < 0.001
end
