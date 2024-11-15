import LinearAlgebra: dot
export LinearModel, DFE
export nearest

### linear system model

"""
    LinearModel(ptype, n)

Linear scalar system model of specified order `n` and type `ptype`, such that:
```
y[t] = dot(p, x[t:-1:t-n+1]) + noise
```
for input `x`, output `y`, parameters `p` and time index `t`.
"""
struct LinearModel{T} <: SystemModel
  n::Int
end

LinearModel(T, p) = LinearModel{T}(p)

function setup(rng, model::LinearModel{T}) where T
  (zeros(T, model.n), zeros(T, model.n))
end

function predict!(model::LinearModel, p, mstate, x)
  circshift!(mstate, 1)
  mstate[1] = x
  dot(p, mstate), mstate
end

### DFE

# struct DFE{T1,T2} <: SystemModel
#   ffsize::Int
#   fbsize::Int
#   decision::T2
# end

# DFE(fbsize::Int, decision) = DFE{ComplexF32,typeof(decision)}(1, fbsize, decision)
# DFE(ffsize::Int, fbsize::Int, decision) = DFE{ComplexF32,typeof(decision)}(ffsize, fbsize, decision)
# DFE(T::DataType, fbsize::Int, decision) = DFE{T,typeof(decision)}(1, fbsize, decision)
# DFE(T::DataType, ffsize::Int, fbsize::Int, decision) = DFE{T,typeof(decision)}(ffsize, fbsize, decision)

# function setup(rng, model::DFE{T1,T2}) where {T1,T2}
#   (zeros(T1, model.ffsize + model.fbsize), zeros(T1, model.fbsize))
# end

# function predict!(model::DFE, ps, st, x, y, t)
#   ffsize = model.ffsize
#   fbsize = model.fbsize
#   if t < ffsize
#     dloss[1:t] .= @view x[t:-1:1]
#     dloss[t+1:ffsize] .= 0
#   else
#     dloss[1:ffsize] .= @view x[t:-1:t-ffsize+1]
#   end
#   dloss[ffsize+1:end] .= st
#   ŷ = dot(ps, dloss)
#   circshift!(st, 1)
#   ȳ = t ≤ length(y) ? y[t] : model.decision(ŷ)
#   st[1] = ȳ
#   e = ȳ - ŷ
#   (ŷ, e)
# end

#   loss = abs2(e)
#   dloss .*= -2 * conj(e)
#   (e, loss, ŷ)
# end

### utilities

"""
    nearest(constellation)

Returns a function that maps a complex number to the nearest value from
a `constellation` set.
"""
function nearest(constellation)
  x -> argmin(y -> abs2(x - y), constellation)
end
