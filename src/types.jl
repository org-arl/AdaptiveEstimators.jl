"Base type for a system model."
abstract type SystemModel end

"Base type for a adaptive parameter estimation algorithm."
abstract type Estimator end

"""
    setup(rng::AbstractRNG, model::SystemModel)

Initializes the model parameters `p` and model state `mstate` using random number
generator `rng`. Parameters `p` are stored as an `AbstractVector` and may be
modified during training by an estimator. Model state `mstate` is opaque and
stores any mutable information needed by the model.

Returns tuple `(p, mstate)`.
"""
function setup end

"""
    setup(rng::AbstractRNG, alg::Estimator, p::AbstractVector)

Initializes the estimator state `estate` using random number generator `rng`
and model parameters `p`. The estimator state `estate` is opaque and stores any
mutable information needed by the estimator.

Returns `estate`.
"""
setup(rng, alg::Estimator, p) = nothing

"""
    predict!(model::SystemModel, p::AbstractVector, mstate, x)

Predicts output `y` and tangent `dy` given input `x`, model parameters `p`,
and model state `mstate`. If the model is a scalar output model, tangent `dy`
is a gradient vector with respect to `p`. If the model is a vector output model,
tangent `dy` is a Jacobian matrix.

The returned `y` and `dy` may be views or cached values that change in-place
at each call for efficiency. The caller should not modify the returned values
or assume that they remain constant after the next call to this function.

Model state `mstate` may be mutated during the call. Input `x` and parameters
`p` are not modified.

Returns tuple `(y, dy)`. For efficiency, `dy` may be a view or a cached value
that changes in-place at each call to `predict!` or `update!`. A copy should
be made if the caller needs to store the tangent for later use.
"""
function predict! end

"""
    update!(model::SystemModel, mstate, y)

Updates the model state `mstate` using desired model output `y`.
"""
update!(model::SystemModel, mstate, y) = nothing

"""
    update!(alg::Estimator, p::AbstractVector, estate, e, dy)

Updates the model parameters `p` using prediction error `e` and tangent `dy`.
The parameters `p` are modified in-place. Estimator state `estate` may be
mutated during the call. Returns loss value.
"""
function update! end
