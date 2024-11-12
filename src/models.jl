using LinearAlgebra: dot

export LinearModel, nearest

### linear system model

"""
    LinearModel()

System model where the `data` is assumed to be of the form `(x, y)` and
parameters `ps` are conjugate impulse response taps. If `p = length(ps)`,
the linear system model is `y[t] = dot(ps, x[t:-1:t-p+1]) + noise`.

If `decision` is specified, it is used to make decisions in decision-directed
mode when `t` exceeds the length of `y`.
"""
struct LinearModel{T} <: SystemModel
  decision::T
end

LinearModel() = LinearModel(nothing)

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
