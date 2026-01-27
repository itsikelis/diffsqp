import torch
from diffsqp.costs import Cost, LqrCost


def test_cost_derivatives(cost: Cost):
    # Helper to compute Jacobian of a gradient w.r.t a variable
    def get_jacobian(grad, var):
        # Flattens the gradient and variable to compute a standard Jacobian matrix
        batch_size = grad.shape[0]
        grad_flat = grad.view(batch_size, -1)
        var_flat = var.view(batch_size, -1)

        jac = []
        for i in range(grad_flat.shape[1]):
            grad_out = torch.zeros_like(grad_flat)
            grad_out[:, i] = 1
            # retain_graph=True allows us to call backward multiple times
            j_col = torch.autograd.grad(
                grad_flat,
                var,
                grad_outputs=grad_out,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if j_col is None:
                j_col = torch.zeros((batch_size, 1, var.shape[1]))
            jac.append(j_col.view(batch_size, 1, -1))
        return torch.cat(jac, dim=1)

    # Create random inputs with gradients enabled
    x = torch.randn(n_batch, n_state, requires_grad=True)
    u = torch.randn(n_batch, n_ctrl, requires_grad=True)

    # Compute analytical values from your class
    lx_analytic = cost.lx(x, u)
    lu_analytic = cost.lu(x, u)
    lxx_analytic = cost.lxx(x, u)
    luu_analytic = cost.luu(x, u)
    lux_analytic = cost.lux(x, u)
    lxu_analytic = cost.lxu(x, u)

    # Compute numerical gradients using autograd
    # First Gradients
    l_sum = cost.l(x, u).sum()
    l_sum.backward(create_graph=True)

    lx_numeric = x.grad
    lu_numeric = u.grad

    lxx_numeric = get_jacobian(x.grad, x)
    luu_numeric = get_jacobian(u.grad, u)
    lxu_numeric = get_jacobian(x.grad, u)
    lux_numeric = get_jacobian(u.grad, x)

    # Assert shapes
    assert lx_analytic.shape == (n_batch, n_state)
    assert lu_analytic.shape == (n_batch, n_ctrl)
    assert lxx_analytic.shape == (n_batch, n_state, n_state)
    assert luu_analytic.shape == (n_batch, n_ctrl, n_ctrl)
    assert lxu_analytic.shape == (n_batch, n_state, n_ctrl)
    assert lux_analytic.shape == (n_batch, n_ctrl, n_state)

    # Final values
    assert torch.allclose(lx_analytic, x.grad)
    assert torch.allclose(lu_analytic, u.grad)
    assert torch.allclose(lxx_analytic, lxx_numeric, atol=1e-6)
    assert torch.allclose(lxu_analytic, lxu_numeric, atol=1e-6)
    assert torch.allclose(lux_analytic, lux_numeric, atol=1e-6)

    print("Cost tests passed!")


if __name__ == "__main__":
    n_batch = 2
    n_state = 3
    n_ctrl = 2
    x_des = torch.randn((n_batch, n_state))
    u_des = torch.randn((n_batch, n_ctrl))
    Q = torch.rand(n_batch, n_state, 1) * torch.eye(n_state).repeat(n_batch, 1, 1)
    R = torch.rand(n_batch, n_ctrl, 1) * torch.eye(n_ctrl).repeat(n_batch, 1, 1)

    cost = LqrCost(Q, R, x_des, u_des)

    test_cost_derivatives(cost)
