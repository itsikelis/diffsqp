import os
import time
import argparse
import torch
import yaml

from dataclasses import dataclass
import numpy as np

from diffsqp.problems import Problem, ProblemParameters
from diffsqp.costs import LqrCost
from diffsqp.solvers import sqp_solve, SqpParameters
from diffsqp.dynamics import Dynamics
from diffsqp.constraints import (
    StateBounds,
    ControlBounds,
)
from diffsqp.types import SqpSolution

from diffsqp.utils.load_save import *


@dataclass
class DoubleIntegratorParameters:
    name: str = "double_integrator"
    n_x: int = 4
    n_q: int = 2
    n_v: int = 2
    n_j: int = 0
    n_u: int = 2
    n_h: int = 0


def main(args):
    device = args.device
    torch.set_default_device(device)
    batch_size = args.batch_size

    sqp_parameters = SqpParameters(
        **{
            ## ADMM ##
            "admm_max_iter": 100,
            "admm_alpha": 1.6,
            "admm_sigma": 1e-6,
            # Rho related
            "admm_reset_rho": False,
            "admm_update_rho": True,
            "admm_rho_init": 150.0,
            "admm_rho_min": 1e-6,
            "admm_rho_max": 1e8,
            "admm_adaptive_rho_tolerance": 10.0,
            "admm_rho_update_iter_freq": 10,
            # Warm starting
            "admm_warm_start_unconstrained": False,
            "admm_reset_ksi": False,
            # Tolerances
            "admm_abs_tolerance": 0.01,
            "admm_abs_tolerance_final": -1.0,
            "admm_rel_tolerance": 0.001,
            "admm_rel_tolerance_final": -1.0,
            "admm_tolerance_update_steps": 0,
            ## SQP ##
            "sqp_max_iter": 3,
            "lqr_reg_init": 1e-5,
            "merit_mu": 1e5,
            "armijo_beta": 1e-3,
            "ls_max_iter": 10,
            "sqp_cost_eps": 1e-1,
            "sqp_viol_eps": 1e-4,
            "qp_solver": "lqr",
            "ls_function": "filter",
        }
    )

    problem_parameters = ProblemParameters(
        **{
            "inverse_dynamics": False,
            "n_h": 0,
            "batch_size": batch_size,
            "dt": 0.01,
            "tf": 1.0,
            "n_h": 0,
            "x_init": [0.0, 0.0, 0.0, 0.0],
            "x_des": [1.0, 0.5, 0.0, 0.0],
            "noise_std": [0.01, 0.01, 0.001, 0.001],
            # State-control bounds
            "x_lb": [-1.0, -1e6, -1e6, -1e6],
            "x_ub": [1.0, 1e6, 1e6, 1e6],
            "u_lb": [-3.0, -3.0],
            "u_ub": [3.0, 3.0],
            # Cost weights
            "q_w": [1e-12, 1e-12, 1e-12, 1e-12],
            "r_w": [1e-2, 1e-2],
            "qf_w": [1e1, 1e1, 1e1, 1e1],
        }
    )

    system_parameters = DoubleIntegratorParameters()

    # print(sqp_parameters)
    print(problem_parameters)
    # print(system_parameters)

    dynamics = Dynamics(nx=4, nu=2, nq=2, nv=2)

    # Create problem
    problem = Problem(problem_parameters, system_parameters)

    # Costs
    Q = problem_parameters.q_w * torch.eye(dynamics.nx).repeat(
        problem_parameters.batch_size, 1, 1
    )
    R = problem_parameters.r_w * torch.eye(dynamics.nu).repeat(
        problem_parameters.batch_size, 1, 1
    )
    Qf = problem_parameters.qf_w * torch.eye(dynamics.nx).repeat(
        problem_parameters.batch_size, 1, 1
    )

    if args.load:
        x, u = load_solution(args.load, device=device)
    else:
        x = torch.zeros((problem.batch_size, problem.horizon, problem.n_x))
        u = torch.zeros((problem.batch_size, problem.horizon - 1, problem.n_u))

    initial_guess = SqpSolution(
        x=x,
        u=u,
        mu=torch.zeros((problem.batch_size, problem.horizon, problem.n_x)),
        nu=torch.zeros((problem.batch_size, problem.horizon - 1, problem.n_h)),
        ksi=[None] * problem.horizon,
    )

    # Randomise Initial state
    initial_guess.x[:, 0] = problem_parameters.x_init.clone()
    noise_std = problem_parameters.noise_std
    noise_dim = len(noise_std)
    initial_guess.x[:, 0] += torch.tensor(noise_std) * torch.randn(
        (batch_size, noise_dim)
    )

    # Set stage costs an initial guess
    for k in range(problem.horizon - 1):
        problem.costs.append([LqrCost(Q=Q, R=R)])
        problem.constraints[k] = [
            StateBounds(
                problem.n_x,
                problem.n_u,
                problem_parameters.x_lb,
                problem_parameters.x_ub,
            ),
            ControlBounds(
                problem.n_x,
                problem.n_u,
                problem_parameters.u_lb,
                problem_parameters.u_ub,
            ),
        ]
    problem.constraints[-1] = [
        StateBounds(
            problem.n_x,
            problem.n_u,
            problem_parameters.x_lb,
            problem_parameters.x_ub,
        )
    ]
    # Set terminal cost
    problem.costs.append(
        [LqrCost(Q=Qf, x_des=problem_parameters.x_des.detach().clone())]
    )

    # Dynamics Constraints
    problem.dynamics = dynamics

    # Solve
    print("Solving Double Integrator Task...")
    solution, log = sqp_solve(problem, sqp_parameters, initial_guess)

    print(log)

    if args.save:
        print(f"Saving solution to {args.save}.pt...")
        save_solution(solution, args.save, x_des=problem_parameters.x_des)
        log.save_to_json(args.save)

    import matplotlib.pyplot as plt
    from diffsqp.utils.plot import plot_trajectories

    plot_trajectories(solution.x, solution.u)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-batch_size", type=int, help="Batch size", default=1)
    parser.add_argument("-device", type=str, help="Batch size", default="cpu")
    parser.add_argument("-save", type=str, help="Filename to save result")
    parser.add_argument("-load", type=str, help="Filename to load result")
    main(parser.parse_args())
