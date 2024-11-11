export LMS, NLMS

### LMS

"""
    LMS(μ=0.01)

Least Mean Squares (LMS) algorithm with step size μ.
"""
struct LMS{T} <: Estimator
  μby2::T
  LMS(μ) = new{typeof(μ)}(μ / 2)
end

LMS() = LMS(1f-2)

Base.show(io::IO, alg::LMS) = print(io, "LMS($(repr(alg.μby2 * 2)))")

function update!(alg::LMS, ps, st, data, loss, dloss, t)
  @. ps -= alg.μby2 * dloss
end

### NLMS

"""
    NLMS(μ=0.1)

Normalized Least Mean Squares (NLMS) algorithm with step size μ.
"""
struct NLMS{T} <: Estimator
  μby2::T
  NLMS(μ) = new{typeof(μ)}(μ / 2)
end

NLMS() = NLMS(1f-1)

Base.show(io::IO, alg::NLMS) = print(io, "NLMS($(repr(alg.μby2 * 2)))")

function update!(alg::NLMS, ps, st, data, loss, dloss, t)
  if t < length(ps)
    sf = @views sum(abs2, data[1][t:-1:1])
  else
    sf = @views sum(abs2, data[1][t:-1:t-length(ps)+1])
  end
  sf += eps(typeof(sf))
  @. ps -= alg.μby2 * dloss / sf
end
