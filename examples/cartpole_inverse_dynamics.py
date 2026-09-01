import os
import sys
import time
import argparse
import torch
import yaml

import numpy as np

from diffsqp.problems import Problem, ProblemParameters
from diffsqp.costs import LqrCost
from diffsqp.solvers import sqp_solve, SqpParameters
from diffsqp.dynamics import Dynamics
from diffsqp.dynamics import CartPoleParameters
from diffsqp.constraints import (
    CartPoleUnderactuation,
    StateBounds,
    ControlBounds,
)
from diffsqp.utils.animate import CartPoleAnimator
from diffsqp.types import SqpSolution

import matplotlib.pyplot as plt


def save_solution(solution: SqpSolution, filepath: str) -> None:
    """Saves only the x and u tensors for warm starting."""
    # Best practice: Move tensors to the CPU before saving.
    # This ensures you won't get CUDA errors if you try to load them on
    # a machine with a different GPU setup or no GPU at all.
    warmstart_data = {"x": solution.x.detach().cpu(), "u": solution.u.detach().cpu()}
    torch.save(warmstart_data, filepath)
    print(f"Warmstart data saved to {filepath}")


def load_solution(
    filepath: str, device: torch.device = torch.device("cpu")
) -> tuple[torch.Tensor, torch.Tensor]:
    """Loads x and u tensors and sends them to the target device."""
    # weights_only=True is recommended in modern PyTorch for security
    # map_location ensures the tensors load directly onto your target device
    data = torch.load(filepath, map_location=device, weights_only=False)

    x = data["x"]
    u = data["u"]

    return x, u


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
    # plt.show()
    plt.savefig("admm.png")


sqp_parameters_dict = {
    ## ADMM ##
    "admm_max_iter": 50,
    "admm_alpha": 1.6,
    "admm_sigma": 1e-6,
    # Rho related
    "admm_reset_rho": False,
    "admm_update_rho": True,
    "admm_rho_init": 0.4,
    "admm_rho_min": 1e-6,
    "admm_rho_max": 1e8,
    "admm_adaptive_rho_tolerance": 10.0,
    "admm_rho_update_iter_freq": 10,
    # Warm starting
    "admm_warm_start_unconstrained": True,
    "admm_reset_ksi": False,
    # "admm_initialize_unconstrained": False,
    # Tolerances
    "admm_abs_tolerance": 0.001,
    "admm_abs_tolerance_final": -1.0,
    "admm_rel_tolerance": 0.0001,
    "admm_rel_tolerance_final": -1.0,
    "admm_tolerance_update_steps": 0,
    ## SQP ##
    "sqp_save_solution": False,
    "sqp_warm_start": False,
    "sqp_warm_start_file_name": "lqr_solution.pt",
    "sqp_max_iter": 100,
    "merit_mu": 1e6,
    "ls_max_iter": 10,
    "sqp_eps": 1e-4,
    "qp_solver": "lqr",
    "ls_function": "filter",
}
sqp_parameters = SqpParameters(**sqp_parameters_dict)

problem_parameters_dict = {
    # "inverse_dynamics": True,
    # "n_h": 1,
    "inverse_dynamics": False,
    "n_h": 0,
    "batch_size": 3,
    "dt": 0.01,
    "tf": 1.0,
    "x_init": [0.0, 0.0, 0.0, 0.0],
    "x_des": [0.0, 3.14159, 0.0, 0.0],
    "noise_std": 0.1,
    # State-control bounds
    "x_lb": [-2.0, -1e6, -5.0, -15.0],
    "x_ub": [2.0, 1e6, 5.0, 15.0],
    "u_lb": [-1e6, -1e6],
    "u_ub": [1e6, 1e6],
    # Cost weights
    "q_w": [1e-6, 1e-6, 1e-6, 1e-6],
    "r_w": [1e-3, 1e-3],
    "qf_w": [1e5, 1e5, 1e5, 1e5],
}
problem_parameters = ProblemParameters(**problem_parameters_dict)

system_parameters_dict = {
    "name": "cartpole",
    "n_x": 4,  # Number of state elements
    "n_q": 2,  # Number of position elements in state
    "n_v": 2,  # Number of velocity elements in state
    "n_j": 2,  # Number of joints
    "n_u": 1,  # Number of DoF
    "mc": 0.5,
    "mp": 0.3,
    "lp": 0.2,
    "grav": 9.81,
}
system_parameters = CartPoleParameters(**system_parameters_dict)

print(sqp_parameters)
print(problem_parameters)
print(system_parameters)

dynamics = Dynamics(
    nx=system_parameters.n_x,
    nu=system_parameters.n_j,  # !!! Underactuated means n_u = n_j
    nq=system_parameters.n_q,
    nv=system_parameters.n_v,
)
underactuation = CartPoleUnderactuation(system_parameters)

# Create problem
print(f"Solving..")
problem = Problem(problem_parameters)

if sqp_parameters.sqp_warm_start:
    target_device = torch.device("cpu")
    x, u = load_solution(sqp_parameters.sqp_warm_start_file_name, device=target_device)
    # u = torch.zeros((problem.batch_size, problem.horizon - 1, problem.n_u))
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

plot_trajectories(initial_guess.x, initial_guess.u)

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

# Set stage costs, constraints and initial guess
for k in range(problem.horizon - 1):
    # initial_guess.x[:, k] = problem_parameters.x_init.detach().clone()
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
        CartPoleUnderactuation(system_parameters),
    ]
# Terminal stage
# initial_guess.x[:, -1] = problem_parameters.x_des.detach().clone()
problem.costs.append([LqrCost(Q=Qf, x_des=problem_parameters.x_des.detach().clone())])
problem.constraints[-1] = [
    StateBounds(
        problem.n_x,
        problem.n_u,
        problem_parameters.x_lb,
        problem_parameters.x_ub,
    )
]


# Dynamics Constraints
problem.dynamics = dynamics
# Underactuation Constraints
if problem_parameters.inverse_dynamics:
    problem.underactuation = underactuation

# Solve
solution, log = sqp_solve(problem, sqp_parameters, initial_guess)

# print("Time elapsed: ", log.solve_wall_time_s, " s.")

print(log)
# log.save_to_json("lqr_solution.json")
if sqp_parameters.sqp_save_solution:
    save_solution(solution, sqp_parameters.sqp_warm_start_file_name)

plot_trajectories(solution.x, solution.u)

# Animate:
animator = CartPoleAnimator(
    solution.x,
    system_parameters.lp,
    problem_parameters.dt,
    problem_parameters.batch_size,
)
animator.animate(step_size=2)
# animator.save(filename="admm.mp4", step_size=2)
