"""Finite-difference checks of the Go2 example derivatives (dynamics and constraints).

    uv run python tests/test_go2_derivatives.py
    uv run --with pytest pytest tests/test_go2_derivatives.py

The examples are loaded by path. Checks run in float64 on the CPU, at random
states near the stance with non-unit quaternions (the constraints see the
normalized one) and non-zero velocities, accelerations and forces.
"""

import importlib.util
import os
from copy import copy as _copy

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DTYPE = torch.float64


def _load_example(name):
    path = os.path.join(ROOT, "examples", f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"example_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _random_point(n=6, seed=0):
    """States and controls (n, 37), (n, 30) near the stance."""
    g = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, generator=g, dtype=DTYPE)

    q = torch.tensor([0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0] + [0.0, 0.8, -1.6] * 4, dtype=DTYPE)
    q = q + 0.1 * rand(n, 19)
    q[:, 3:7] *= 1.0 + 0.1 * rand(n, 1).abs()
    v = rand(n, 18)
    a = 3.0 * rand(n, 18)
    f = 20.0 * rand(n, 12)
    return torch.cat([q, v], dim=-1), torch.cat([a, f], dim=-1)


def _fd(fn, z, eps=1e-6):
    """Central-difference Jacobian of fn with respect to the last dimension of z."""
    cols = []
    for i in range(z.shape[-1]):
        dz = torch.zeros_like(z)
        dz[..., i] = eps
        cols.append((fn(z + dz) - fn(z - dz)) / (2.0 * eps))
    return torch.stack(cols, dim=-1)


def _check(name, analytic, fd):
    scale = fd.abs().max().item()
    err = (analytic - fd).abs().max().item()
    assert err <= 1e-6 * max(scale, 1.0), f"{name}: |analytic - fd| = {err:.2e} (scale {scale:.2e})"


def _check_constraint(name, c, x, u, has_u=True):
    _check(f"{name} gx", c.gx(x, u), _fd(lambda z: c.g(z, u), x))
    if has_u:
        _check(f"{name} gu", c.gu(x, u), _fd(lambda z: c.g(x, z), u))


def _check_example(module_name):
    ex = _load_example(module_name)
    model = ex.gdyn.load_urdf(ex.URDF_PATH)
    x, u = _random_point()
    dt = 0.02

    q = ex.normalized(x[:, :19])
    eye = torch.eye(18, dtype=DTYPE).expand(x.shape[0], 18, 18)
    _check("tangent_map @ coordinate_map", ex.tangent_map(q) @ ex.coordinate_map(q), eye)

    dyn = ex.Go2Dynamics(model)
    _check("fx", dyn.fx(x, u, dt), _fd(lambda z: dyn.f(z, u, dt), x))
    _check("fu", dyn.fu(x, u, dt), _fd(lambda z: dyn.f(x, z, dt), u))

    tau_max = torch.ones(12, dtype=DTYPE)
    if hasattr(ex, "Go2BaseUnderactuation"):
        h = ex.Go2BaseUnderactuation(model)
        _check("hx", h.hx(x, u), _fd(lambda z: h.h(z, u), x))
        _check("hu", h.hu(x, u), _fd(lambda z: h.h(x, z), u))
        _check_constraint("JointTorqueBounds", ex.JointTorqueBounds(model, -tau_max, tau_max), x, u)
    else:
        _check_constraint(
            "InverseDynamicsBounds", ex.InverseDynamicsBounds(model, -tau_max, tau_max), x, u
        )

    p0 = torch.zeros(x.shape[0], 4, 3, dtype=DTYPE)
    _check_constraint("FootContact", ex.FootContact(model, p0), x, u)
    cone = ex.FrictionCone(37, 30, mu=0.8)
    cone.G = cone.G.to(DTYPE)
    _check_constraint("FrictionCone", cone, x, u)


def _check_stage_batching(module_name):
    """Problem.linearize and padded_g give the same matrices whether the shared
    constraints run once on (S, B, n) inputs, stage by stage, or the old way."""
    ex = _load_example(module_name)
    model = ex.gdyn.load_urdf(ex.URDF_PATH)
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(DTYPE)
    try:
        torch.manual_seed(0)
        setup = ex.build_problem(model, batch_size=3, horizon=6)
        problem, guess = setup.problem, setup.initial_guess
        guess.x = guess.x + 1e-2 * torch.randn_like(guess.x)
        guess.u = guess.u + 1e-1 * torch.randn_like(guess.u)

        def matrices():
            mat = problem.linearize(guess)
            return mat.M, mat.N, mat.n, problem.padded_g(guess.x, guess.u)

        assert problem._shared_stage_constraints() is problem.constraints[0]
        batched = matrices()
        for c in problem.constraints[0]:
            c.stage_batched = False
        stacked = matrices()

        # Stages that do not share their constraints take the per-stage loop
        problem.constraints[2] = [*problem.constraints[2]]
        problem.constraints[2][-1] = _copy(problem.constraints[2][-1])
        assert problem._shared_stage_constraints() is None
        looped = matrices()
    finally:
        torch.set_default_dtype(default_dtype)

    for name, a, b, c in zip("M N n g".split(), batched, stacked, looped):
        torch.testing.assert_close(a, b, rtol=1e-13, atol=1e-13, msg=f"{name}: batched vs stacked")
        torch.testing.assert_close(a, c, rtol=1e-13, atol=1e-13, msg=f"{name}: batched vs loop")


def test_go2_riccati_derivatives():
    _check_example("go2_inverse_riccati")


def test_go2_admm_derivatives():
    _check_example("go2_inverse_admm")


def test_go2_riccati_stage_batching():
    _check_stage_batching("go2_inverse_riccati")


def test_go2_admm_stage_batching():
    _check_stage_batching("go2_inverse_admm")


def test_mpc_shift_keeps_terminal_rows():
    mpc = _load_example("go2_mpc")
    ksi = torch.arange(5.0).reshape(1, 5, 1)
    shifted = mpc.shift_stages(ksi, 2)
    assert shifted.flatten().tolist() == [2.0, 3.0, 3.0, 3.0, 4.0]


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, fn in tests:
        print(f"{name} ...", flush=True)
        fn()
        print("  ok")
