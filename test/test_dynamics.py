import torch
from diffsqp.dynamics import Dynamics, PendulumDynamics


def test_dynamics_derivatives(
    dyn: Dynamics, x: torch.Tensor, u: torch.Tensor, dt: float
):
    m = dyn.m
    l = dyn.l
    b = dyn.b
    dt = 0.01
    n_batch = x.shape[0]

    # 1. Get analytical Jacobian
    fx_analytic = dyn.fx(x, u, dt)
    fu_analytic = dyn.fu(x, u, dt)

    # 2. Get numerical/autograd Jacobian
    # We define a wrapper to handle the batch dimension for autograd
    def fx_wrapper(x_in):
        return dyn.f(x_in, u, dt)

    def fu_wrapper(u_in):
        return dyn.f(x, u_in, dt)

    # Compute Jacobian for each element in the batch
    fx_numeric = []
    fu_numeric = []
    for i in range(n_batch):
        # jacobian returns (output_dim, input_dim)
        jx = torch.autograd.functional.jacobian(fx_wrapper, x)[i, :, i, :]
        ju = torch.autograd.functional.jacobian(fu_wrapper, u)[i, :, i, :]
        fu_numeric.append(ju)
        fx_numeric.append(jx)
    fx_numeric = torch.stack(fx_numeric)
    fu_numeric = torch.stack(fu_numeric)

    # 3. Compare
    assert torch.allclose(fx_analytic, fx_numeric, atol=1e-6)
    assert torch.allclose(fu_analytic, fu_numeric, atol=1e-6)

    print("Dynamics tests passed!")


if __name__ == "__main__":
    n_batch = 2
    m, l, b = 1.0, 1.0, 0.1
    dt = 0.01
    dyn = PendulumDynamics(m, l, b)

    x = torch.randn(n_batch, 2, requires_grad=True)
    u = torch.randn(n_batch, 1, requires_grad=True)

    test_dynamics_derivatives(dyn, x, u, dt)
