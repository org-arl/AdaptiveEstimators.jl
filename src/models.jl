import LinearAlgebra: dot
export LinearModel, nearest

### linear system model

"""
    LinearModel(ptype, p)
    LinearModel(ptype, p, decision)

Linear system model of specified order `p` and parameters `ps` of type `ptype`.
The `data` is assumed to be of the form `(x, y)` such that:
`y[t] = dot(ps, x[t:-1:t-p+1]) + noise`.

If `decision` is specified, it is used to make decisions in decision-directed
mode when `t` exceeds the length of `y`.
"""
struct LinearModel{T1,T2} <: SystemModel
  p::Int
  decision::T2
end

LinearModel(T, p) = LinearModel{T,typeof(identity)}(p, identity)
LinearModel(T, p, decision) = LinearModel{T,typeof(decision)}(p, decision)

function setup(rng, model::LinearModel{T1,T2}) where {T1,T2}
  (zeros(T1, model.p), nothing)
end

function loss_and_gradient!(model::LinearModel, dloss, ps, st, (x, y), t)
  if t < length(ps)
    dloss[1:t] .= @view x[t:-1:1]
    dloss[t+1:end] .= 0
  else
    dloss .= @view x[t:-1:t-length(ps)+1]
  end
  ŷ = dot(ps, dloss)
  ȳ = t ≤ length(y) ? y[t] : model.decision(ŷ)
  e = ȳ - ŷ
  loss = abs2(e)
  dloss .*= -2 * conj(e)
  (loss, st, ŷ)
end

### utilities

"""
    nearest(constellation)

Returns a function that maps a complex number to the nearest value from
a `constellation` set.
"""
function nearest(constellation)
  x -> argmin(y -> abs2(x - y), constellation)
end
