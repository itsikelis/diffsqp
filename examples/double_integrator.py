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


def plot_trajectories(states_tensor, controls_tensor):
    # Detach and convert to numpy for the first batch
    states_np = states_tensor[0, :, :].detach().cpu().numpy()
    controls_np = controls_tensor[0, :, :].detach().cpu().numpy()

    horizon_x, n_x = states_np.shape
    horizon_u, n_u = controls_np.shape

    time_x = range(horizon_x)
    time_u = range(horizon_u)

    # Create a figure with 2 vertically stacked subplots
    # sharex=True aligns the time steps on the x-axis for both plots
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- Top Subplot: States ---
    for i in range(n_x):
        axs[0].plot(time_x, states_np[:, i], label=f"State $x_{{{i}}}$")

    axs[0].set_ylabel("Value")
    axs[0].set_title("State Trajectory (First Environment)")
    axs[0].legend()
    axs[0].grid(True)

    # --- Bottom Subplot: Controls ---
    for i in range(n_u):
        axs[1].plot(time_u, controls_np[:, i], label=f"Control $u_{{{i}}}$")

    axs[1].set_xlabel("Time Step $k$")
    axs[1].set_ylabel("Value")
    axs[1].set_title("Control Trajectory (First Environment)")
    axs[1].legend()
    axs[1].grid(True)

    # Adjust layout to prevent overlap and display
    plt.tight_layout()
    plt.show()
    # plt.savefig("constrained.png")


plot_trajectories(solution.x, solution.u)

exit()

# Animate:
if sys_params.name == "acrobot":
    anim = AcrobotAnimator(
        solution.x,
        sys_params.l1,
        sys_params.l2,
        problem_parameters.dt,
        problem_parameters.batch_size,
    )
elif sys_params.name == "cartpole":
    anim = CartPoleAnimator(
        solution.x,
        sys_params.lp,
        problem_parameters.dt,
        problem_parameters.batch_size,
    )

anim.animate(step_size=2)
# anim.save(filename="four_batches.mp4", step_size=2)
