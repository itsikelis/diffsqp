"""Small trajectory optimization problems for solver tests and benchmarks.

Each builder returns ``(problem, initial_guess, sqp_args)``. ``sqp_args`` is a
plain dict of ``SqpParameters`` keyword arguments, so callers can override
entries before building the parameters:

    problem, guess, args = cartpole_forward(batch_size=8, seed=0)
    args["ls_function"] = "merit"
    solution, log = sqp_solve(problem, SqpParameters(**args), guess)

All problems start from a zero trajectory with a randomly perturbed initial
state per batch element (seeded, so runs are reproducible).
"""

import torch

from diffsqp.problems import Problem, ProblemParameters
from diffsqp.costs import LqrCost
from diffsqp.dynamics import Dynamics, CartPoleDynamics, CartPoleParameters
from diffsqp.constraints import CartPoleUnderactuation, StateBounds, ControlBounds
from diffsqp.types import SqpSolution


def base_sqp_args(**overrides):
    args = {
        ## ADMM ##
        "admm_max_iter": 30,
        "admm_alpha": 1.6,
        "admm_sigma": 1e-6,
        # Rho related
        "admm_reset_rho": False,
        "admm_update_rho": False,
        "admm_rho_init": 0.1,
        "admm_rho_min": 1e-6,
        "admm_rho_max": 1e8,
        "admm_adaptive_rho_tolerance": 10.0,
        "admm_rho_update_iter_freq": 30,
        # Warm starting
        "admm_warm_start_unconstrained": True,
        "admm_reset_ksi": False,
        # Tolerances
        "admm_abs_tolerance": 0.001,
        "admm_abs_tolerance_final": -1.0,
        "admm_rel_tolerance": 0.0001,
        "admm_rel_tolerance_final": -1.0,
        "admm_tolerance_update_steps": 0,
        ## SQP ##
        "sqp_max_iter": 50,
        "lqr_reg_init": 1e-5,
        "merit_mu": 1e7,
        "armijo_beta": 1e-3,
        "ls_max_iter": 10,
        "sqp_cost_eps": 1e-1,
        "sqp_viol_eps": 1e-4,
        "check_complementarity": False,
        "qp_solver": "lqr",
        "ls_function": "filter",
        "verbose": False,
    }
    args.update(overrides)
    return args


def _zero_guess(problem, x_init, noise_std, seed, x_lb=None, x_ub=None):
    batch_size = problem.batch_size
    guess = SqpSolution(
        x=torch.zeros((batch_size, problem.horizon, problem.n_x)),
        u=torch.zeros((batch_size, problem.horizon - 1, problem.n_u)),
        mu=torch.zeros((batch_size, problem.horizon, problem.n_x)),
        nu=torch.zeros((batch_size, problem.horizon - 1, problem.n_h)),
        ksi=[None] * problem.horizon,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(
        (batch_size, len(noise_std)), generator=generator, device="cpu"
    )
    x0 = x_init + torch.tensor(noise_std) * noise.to(guess.x.device)
    if x_lb is not None:
        # Keep the start strictly inside the state bounds: the initial state is
        # fixed, so a violated stage-0 bound can never be repaired
        x0 = torch.clamp(x0, 0.9 * x_lb, 0.9 * x_ub)
    guess.x[:, 0] = x0
    return guess


def _add_lqr_costs_and_bounds(problem, params, n_x, n_u):
    batch_size = problem.batch_size
    Q = params.q_w * torch.eye(n_x).repeat(batch_size, 1, 1)
    R = params.r_w * torch.eye(n_u).repeat(batch_size, 1, 1)
    Qf = params.qf_w * torch.eye(n_x).repeat(batch_size, 1, 1)

    state_bounds = StateBounds(problem.n_x, problem.n_u, params.x_lb, params.x_ub)
    control_bounds = ControlBounds(problem.n_x, problem.n_u, params.u_lb, params.u_ub)

    for k in range(problem.horizon - 1):
        problem.costs.append([LqrCost(Q=Q, R=R)])
        problem.constraints[k] = [state_bounds, control_bounds]
    problem.costs.append([LqrCost(Q=Qf, x_des=params.x_des.clone())])
    problem.constraints[-1] = [state_bounds]


def double_integrator(batch_size=4, tf=1.0, seed=0, noise_scale=1.0):
    """Planar point mass reaching (1, 0.5) with |x| <= 1.2 and |u| <= 10.

    The dynamics are linear and the cost is quadratic, so this is a single
    convex QP: SQP should finish in one or two iterations.
    """
    params = ProblemParameters(
        inverse_dynamics=False,
        n_h=0,
        batch_size=batch_size,
        dt=0.01,
        tf=tf,
        x_init=[0.0, 0.0, 0.0, 0.0],
        x_des=[1.0, 0.5, 0.0, 0.0],
        noise_std=[0.01 * noise_scale, 0.01 * noise_scale, 1e-3, 1e-3],
        x_lb=[-1.2, -1e6, -1e6, -1e6],
        x_ub=[1.2, 1e6, 1e6, 1e6],
        u_lb=[-10.0, -10.0],
        u_ub=[10.0, 10.0],
        q_w=[1e-6, 1e-6, 1e-6, 1e-6],
        r_w=[1e-2, 1e-2],
        qf_w=[1e3, 1e3, 1e3, 1e3],
    )

    class _Sys:
        n_x, n_u = 4, 2

    problem = Problem(params, _Sys)
    problem.dynamics = Dynamics(nx=4, nu=2, nq=2, nv=2)
    _add_lqr_costs_and_bounds(problem, params, 4, 2)

    guess = _zero_guess(
        problem, params.x_init, params.noise_std, seed, params.x_lb, params.x_ub
    )
    args = base_sqp_args(
        admm_max_iter=100,
        admm_rho_init=150.0,
        admm_update_rho=True,
        admm_rho_update_iter_freq=10,
        admm_abs_tolerance=0.01,
        admm_rel_tolerance=0.001,
        admm_warm_start_unconstrained=False,
        merit_mu=1e5,
        sqp_max_iter=5,
    )
    return problem, guess, args


def _cartpole_system(n_u):
    return CartPoleParameters(
        name="cartpole",
        n_x=4,
        n_q=2,
        n_v=2,
        n_j=2,
        n_u=n_u,
        mc=0.5,
        mp=0.3,
        lp=0.2,
        grav=9.81,
    )


def cartpole_forward(batch_size=4, tf=1.0, seed=0, noise_scale=1.0, u_max=20.0):
    """Cart-pole swing-up with nonlinear forward dynamics and box bounds."""
    params = ProblemParameters(
        inverse_dynamics=False,
        n_h=0,
        batch_size=batch_size,
        dt=0.01,
        tf=tf,
        x_init=[0.0, 0.0, 0.0, 0.0],
        x_des=[0.0, 3.14159, 0.0, 0.0],
        noise_std=[0.01 * noise_scale, 0.01 * noise_scale, 1e-3, 1e-3],
        x_lb=[-2.5, -1e6, -5.0, -20.0],
        x_ub=[2.5, 1e6, 5.0, 20.0],
        u_lb=[-u_max],
        u_ub=[u_max],
        q_w=[1e-1, 1e-1, 1e-1, 1e-1],
        r_w=[1e0],
        qf_w=[1e6, 1e6, 1e6, 1e6],
    )
    system = _cartpole_system(n_u=1)
    problem = Problem(params, system)
    problem.dynamics = CartPoleDynamics(system)
    _add_lqr_costs_and_bounds(problem, params, 4, 1)

    guess = _zero_guess(
        problem, params.x_init, params.noise_std, seed, params.x_lb, params.x_ub
    )
    return problem, guess, base_sqp_args()


def cartpole_inverse(batch_size=4, tf=1.0, seed=0, noise_scale=1.0):
    """Cart-pole swing-up with inverse dynamics and an underactuation constraint."""
    params = ProblemParameters(
        inverse_dynamics=True,
        n_h=1,
        batch_size=batch_size,
        dt=0.01,
        tf=tf,
        x_init=[0.0, 0.0, 0.0, 0.0],
        x_des=[0.0, 3.14159, 0.0, 0.0],
        noise_std=[0.1 * noise_scale, 0.01 * noise_scale, 1e-3, 1e-3],
        x_lb=[-2.5, -1e6, -5.0, -20.0],
        x_ub=[2.5, 1e6, 5.0, 20.0],
        u_lb=[-300.0, -300.0],
        u_ub=[300.0, 300.0],
        q_w=[1e-6, 1e-6, 1e-6, 1e-6],
        r_w=[1e-2, 1e-2],
        qf_w=[1e5, 1e5, 1e5, 1e5],
    )
    system = _cartpole_system(n_u=2)
    problem = Problem(params, system)
    problem.dynamics = Dynamics(nx=4, nu=2, nq=2, nv=2)
    problem.underactuation = CartPoleUnderactuation(system)
    _add_lqr_costs_and_bounds(problem, params, 4, 2)

    guess = _zero_guess(
        problem, params.x_init, params.noise_std, seed, params.x_lb, params.x_ub
    )
    return problem, guess, base_sqp_args(admm_update_rho=True, sqp_max_iter=60)


PROBLEMS = {
    "double_integrator": double_integrator,
    "cartpole_forward": cartpole_forward,
    "cartpole_inverse": cartpole_inverse,
}
