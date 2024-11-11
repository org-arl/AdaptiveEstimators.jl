using LinearAlgebra: dot
export LinearModel

"""
    LinearModel()

System model where the `data` is assumed to be of the form `(x, y)` and
parameters `ps` are conjugate impulse response taps. If `p = length(ps)`,
the linear system model is `y[t] = dot(ps, x[t:-1:t-p+1]) + noise`.
"""
struct LinearModel <: SystemModel end

function loss_and_gradient!(::LinearModel, dloss, ps, st, (x, y), t)
  if t < length(ps)
    dloss[1:t] .= @view x[t:-1:1]
    dloss[t+1:end] .= 0
  else
    dloss .= @view x[t:-1:t-length(ps)+1]
  end
  ŷ = dot(ps, dloss)
  e = y[t] - ŷ
  loss = abs2(e)
  dloss .*= -2 * conj(e)
  (loss, st, ŷ)
end
