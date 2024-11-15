import Random
export fit!

"Adaptive estimation results."
struct EstimationResults{T1,T2}
  p::Vector{T1}
  phist::Matrix{T1}
  y::T2
  loss::Vector{Float64}
end

Base.show(io::IO, res::EstimationResults) = print(io, "EstimationResults(p, phist, y, loss)")

"""
    fit!(model::SystemModel, alg::Estimator, x, y; saveat, rng)
    fit!(model::SystemModel, alg::Estimator, x, y, nsteps, decision; ...)

Estimate `model` parameters using `alg` adaptive algorithm. Input `x` and
desired output `y` are vectors (scalar input/output) or matrices (vector
input/output) with the last dimension being the time dimension. Input
`x` is assumed to be available at all time indices `1:nsteps`. Desired
output `y` may be available at all time indices or for a shorter time.
If it is available for a shorter time, the estimator will use a `decision`
function to generated desired output in a decision-directed mode for the
rest of the time.

The `saveat` argument specifies how often to record parameter history.
Random number generator `rng` is used to initialize model and estimator.

Returns `EstimationResults` containing estimated paramaters `p`, parameter
history `phist`, model outputs `y` and loss history `loss`.
"""
function fit!(model::SystemModel, alg::Estimator, x, y,
    nsteps=_len(x), decision=identity; saveat=0, rng=Random.GLOBAL_RNG)
  p, mstate = setup(rng, model)
  estate = setup(rng, alg, p)
  loss = Array{Float64}(undef, nsteps)
  histlen = saveat > 0 ? nsteps ÷ saveat : 0
  phist = similar(p, length(p), histlen)
  out = similar(y, size(y)[1:end-1]..., nsteps)
  for i ∈ 1:nsteps
    out[i], dy = predict!(model, p, mstate, _get(x, i))
    d = i ≤ _len(y) ? _get(y, i) : decision(out[i])
    loss[i] = update!(alg, p, estate, d - out[i], dy)
    saveat > 0 && i % saveat == 0 && (phist[:, i ÷ saveat] .= p)
    update!(model, mstate, d)
  end
  EstimationResults(p, phist, out, loss)
end

# helpers
_get(x::AbstractVector, i) = x[i]
_get(x::AbstractMatrix, i) = x[:,i]
_len(x::AbstractVector) = length(x)
_len(x::AbstractMatrix) = size(x, 2)
