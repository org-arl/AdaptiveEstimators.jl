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

"""
    DFE(ffsize, fbsize)
    DFE(T, ffsize, fbsize)

Decision feedback equalizer with feedforward filter of size `ffsize` and
feedback filter of size `fbsize`. If type `T` is not specified, the filters
are assumed to be `ComplexF32`.
"""
struct DFE{T} <: SystemModel
  ffsize::Int
  fbsize::Int
end

DFE(fbsize::Int) = DFE{ComplexF32}(1, fbsize)
DFE(ffsize::Int, fbsize::Int) = DFE{ComplexF32}(ffsize, fbsize)
DFE(T::DataType, fbsize::Int) = DFE{T}(1, fbsize)
DFE(T::DataType, ffsize::Int, fbsize::Int) = DFE{T}(ffsize, fbsize)

function setup(rng, model::DFE{T}) where T
  (zeros(T, model.ffsize + model.fbsize), zeros(T, model.ffsize + model.fbsize))
end

function predict!(model::DFE, p, mstate, x)
  fb = @views dot(p[model.ffsize+1:end], mstate[model.ffsize+1:end])
  mstate[2:model.ffsize] .= mstate[1:model.ffsize-1]
  mstate[1] = x
  ŷ = @views dot(p[1:model.ffsize], mstate[1:model.ffsize]) + fb
  ŷ, mstate
end

function update!(model::DFE, mstate, y)
  mstate[model.ffsize+2:end] .= mstate[model.ffsize+1:end-1]
  mstate[model.ffsize+1] = y
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
