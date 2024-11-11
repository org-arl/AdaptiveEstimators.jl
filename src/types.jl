"Base type for a system model."
abstract type SystemModel end

"Base type for a adaptive parameter estimation algorithm."
abstract type Estimator end

"""
    loss_and_gradient!(model::SystemModel, dloss::AbstractVector, ps::AbstractVector, st, data, t::Int)

Computes a `loss` and its gradient `dloss` with respect to model parameters `ps`.
`t` is a time index, typically starting at 1 and increasing by 1 each time this
function is called. `st` is the state of the model, and an updated `st` is returned.
`data` contains static information needed by the model.

Array `dloss` is pre-allocated and should be filled with the gradient of the loss.
The size of `dloss` is equal to the size of `ps`.

Returns tuple `(loss, st, out)` where `out` contains any additional information
generated by the model.
"""
function loss_and_gradient! end

"""
    update!(alg::Estimator, ps::AbstractVector, st, data, loss, dloss, t::Int)

Updates the model parameters `ps` using the `loss` and its gradient `dloss`.
Other arguments `t`, `st`, and `data` are the same as in `loss_and_gradient()`.
`loss` and `dloss` are the loss and its gradient computed by `loss_and_gradient()`.
The parameters `ps` are modified in-place and returned.
"""
function update! end
