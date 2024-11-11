export estimate!
export linear_estimate!

"Adaptive estimation results."
struct EstimationResults{T1,T2}
  ps::Vector{T1}
  ps_history::Matrix{T1}
  ts::StepRange{Int,Int}
  out::T2
  loss::Vector{Float64}
end

Base.show(io::IO, res::EstimationResults) = print(io, "EstimationResults(ps, ps_history, ts, out, loss)")

"""
    estimate!(model::SystemModel, alg::Estimator, ps, st, data, ts; saveat=0)

Estimate parameters `ps` of `model` using `alg` adaptive algorithm. Initial
parameters are `ps` and passed in and updated in-place. `st` is the initial
state. `data` contains data in a model-dependent form. `ts` is a list of time
indices. The `saveat` parameter specifies how often to save the parameters
in `ps_history`.

Returns `EstimationResults` containing estimated paramaters `ps`, parameter
history `ps_history`, time indices `ts` corresponding to the history, model
outputs `out` and loss history `loss`.
"""
function estimate!(model::SystemModel, alg::Estimator, ps, st, data, ts; saveat=0)
  dloss = similar(ps)
  loss = Array{Float64}(undef, length(ts))
  histlen = saveat > 0 ? length(ts) ÷ saveat : 0
  ps_history = similar(ps, length(ps), histlen)
  loss[1], st, rv = loss_and_gradient!(model, dloss, ps, st, data, ts[1])
  out = isnothing(rv) ? nothing : Array{typeof(rv)}(undef, length(ts))
  out[1] = rv
  for i ∈ 2:length(ts)
    loss[i], st, out[i] = loss_and_gradient!(model, dloss, ps, st, data, ts[i])
    update!(alg, ps, st, data, loss[i], dloss, ts[i])
    saveat > 0 && i % saveat == 0 && (ps_history[:, i ÷ saveat] .= ps)
  end
  ts_history = saveat > 0 ? ts[saveat:saveat:end] : 1:1:0
  EstimationResults(ps, ps_history, ts_history, out, loss)
end

"""
    linear_estimate!(alg::Estimator, p, x, y, ts=1:length(y); saveat=0)

Convienence function for estimating linear model parameters using algorithm
`alg`. The `p` parameter specifies the length of the model parameter vector
`ps`. The linear system model is `y[t] = dot(ps, x[t:-1:t-p+1]) + noise`.

For more details, see [`estimate!`](@ref).
"""
function linear_estimate!(alg::Estimator, p, x, y, ts=1:length(y); saveat=0)
  estimate!(LinearModel(), alg, zeros(eltype(y), p), nothing, (x, y), ts; saveat)
end
