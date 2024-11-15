import LinearAlgebra: diagm
export LMS, NLMS

### LMS

"""
    LMS(μ=0.01)

Least Mean Squares (LMS) algorithm with step size μ.
"""
struct LMS{T} <: Estimator
  μ::T
end

LMS() = LMS(1f-2)

Base.show(io::IO, alg::LMS) = print(io, "LMS($(repr(alg.μ)))")

function update!(alg::LMS, p, estate, e, dy)
  @. p += alg.μ * dy * conj(e)
  abs2(e)
end

### NLMS

"""
    NLMS(μ=0.1)

Normalized least Mean Squares (NLMS) algorithm with step size μ.
"""
struct NLMS{T} <: Estimator
  μ::T
end

NLMS() = NLMS(1f-1)

Base.show(io::IO, alg::NLMS) = print(io, "NLMS($(repr(alg.μ)))")

function update!(alg::NLMS, p, estate, e, dy)
  sf = sum(abs2, dy)
  sf += eps(typeof(sf))
  @. p += alg.μ * dy * conj(e) / sf
  abs2(e)
end

# ### RLS

# """
#     RLS(λ=0.99, σ=1.0)

# Recursive Least Squares (RLS) algorithm with forgetting factor λ and initial
# covariance σ.
# """
# struct RLS{T} <: Estimator
#   λ::T
#   σ::T
# end

# RLS() = RLS(0.99f0, 1f0)
# RLS(λ) = RLS(λ, one(typeof(λ)))

# Base.show(io::IO, alg::RLS) = print(io, "RLS($(alg.λ), $(alg.σ)")

# setup(rng, alg::RLS, ps) = diagm(ones(eltype(ps), length(ps)) * alg.σ)

# function update!(alg::NLMS, ps, st, data, e, loss, dloss, t)
#   #x?
#   k = st * x ./ (alg.λ + x' * st * x)
#   @. ps += k * e
# end
