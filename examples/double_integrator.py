import os
import time
import argparse
import torch
import yaml

import numpy as np

from diffsqp.problems import Problem, ProblemParameters
from diffsqp.costs import LqrCost
from diffsqp.solvers import sqp_solve, SqpParameters
from diffsqp.dynamics import Dynamics, AcrobotDynamics, CartPoleDynamics
from diffsqp.dynamics import AcrobotParameters, CartPoleParameters
from diffsqp.constraints import (
    AcrobotUnderactuation,
    CartPoleUnderactuation,
    StateBounds,
    ControlBounds,
)
from diffsqp.utils.plot import plot_trajectories
from diffsqp.utils.animate import AcrobotAnimator, CartPoleAnimator
from diffsqp.types import SqpSolution

sqp_parameters_dict = {
    "admm_max_iter": 50,
    "admm_eps": 0.01,
    "admm_alpha": 1.6,
    "admm_sigma": 1e-6,
    "admm_rho_ineq": 0.8,
    "admm_rho_eq": 1000.0,
    "admm_warm_start": True,
    "admm_initialize_unconstrained": True,
    "sqp_max_iter": 100,
    "merit_mu": 1e6,
    "ls_max_iter": 10,
    "sqp_eps": 1e-4,
    "qp_solver": "lqr",
    "ls_function": "filter",
}
sqp_parameters = SqpParameters(**sqp_parameters_dict)

problem_parameters_dict = {
    "inverse_dynamics": False,
    "batch_size": 3,
    "tf": 1.0,
    "dt": 0.01,
    "n_h": 0,
    "x_init": [0.0, 0.0, 0.0, 0.0],
    "noise_std": 0.0,
    "x_des": [1.0, 0.0, 0.0, 0.0],
    # State-control bounds
    "x_lb": [-1e6, -1e6, -1e6, -1e6],
    "x_ub": [1e6, 1e6, 1e6, 1e6],
    "u_lb": [-6.0, -6.0],
    "u_ub": [6.0, 6.0],
    # Cost weights
    "q_w": [1e-6, 1e-6, 1e-6, 1e-6],
    "r_w": [1e-8, 1e-8],
    "qf_w": [1e5, 1e5, 1e5, 1e5],
}
problem_parameters = ProblemParameters(**problem_parameters_dict)

dynamics = Dynamics(nx=4, nu=2, nq=2, nv=2)

# Create problem
print(f"Solving..")
problem = Problem(problem_parameters)
initial_guess = SqpSolution(
    x=torch.zeros((problem.batch_size, problem.horizon, problem.n_x)),
    u=torch.zeros((problem.batch_size, problem.horizon - 1, problem.n_u)),
    mu=torch.zeros((problem.batch_size, problem.horizon, problem.n_x)),
    nu=torch.zeros((problem.batch_size, problem.horizon - 1, problem.n_h)),
    ksi=[None] * problem.horizon,
)

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

# Set stage costs an initial guess
for k in range(problem.horizon - 1):
    initial_guess.x[:, k] = problem_parameters.x_init.clone()
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
# Set terminal cost
initial_guess.x[:, -1] = problem_parameters.x_des.detach().clone()
problem.costs.append([LqrCost(Q=Qf, x_des=problem_parameters.x_des.detach().clone())])

# Dynamics Constraints
problem.dynamics = dynamics

# Solve
solution, log = sqp_solve(problem, sqp_parameters, initial_guess)

print("Time elapsed: ", log.solve_wall_time_s, " s.")

import matplotlib.pyplot as plt


def plot_states(states_tensor):
    first_batch = states_tensor[0, :, :].detach().cpu().numpy()

    horizon, n_x = first_batch.shape
    time = range(horizon)

    # 3. Plot each dimension of the state
    for i in range(n_x):
        plt.plot(time, first_batch[:, i], label=f"State $x_{{{i}}}$")

    plt.xlabel("Time Step $k$")
    plt.ylabel("Value")
    plt.title("State Trajectory (First Environment)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# def plot_controls(controls_tensor):
#     first_batch = controls_tensor[0, :, :].detach().cpu().numpy()
#
#     horizon, n_x = first_batch.shape
#     time = range(horizon)
#
#     # 3. Plot each dimension of the state
#     for i in range(n_x):
#         plt.plot(time, first_batch[:, i], label=f"State $x_{{{i}}}$")
#
#     plt.xlabel("Time Step $k$")
#     plt.ylabel("Value")
#     plt.title("State Trajectory (First Environment)")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     # plt.savefig("state_trajectory.png")
#     plt.show()
#
#
# plot_states(solution.x)
# plot_controls(solution.u)

plot_trajectories(solution.x, solution.u)
