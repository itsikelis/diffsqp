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


def load_config(config_path):
    if not os.path.exists(config_path):
        print(f"Error: Configuration file '{config_path}' not found.")
        sys.exit(1)
    with open(config_path, "r") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(f"Error parsing YAML file: {exc}")
            sys.exit(1)
    return data


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, help="Experiment config file")
args = parser.parse_args()

print(f"Loading problem configuration from: {args.config}")
cfg = load_config(args.config)
print(f"Successfully loaded parameters:")
sqp_params = SqpParameters(**cfg["solver"])
prob_params = ProblemParameters(**cfg["problem"])

if cfg["system"]["name"] == "acrobot":
    sys_params = AcrobotParameters(**cfg["system"])

    if prob_params.inverse_dynamics:
        dyn = Dynamics(
            nx=sys_params.n_x, nu=sys_params.n_j, nq=sys_params.n_q, nv=sys_params.n_v
        )
        uact = AcrobotUnderactuation(sys_params)
    else:
        dyn = AcrobotDynamics(sys_params)

elif cfg["system"]["name"] == "cartpole":
    sys_params = CartPoleParameters(**cfg["system"])

    if prob_params.inverse_dynamics:
        dyn = Dynamics(
            nx=sys_params.n_x, nu=sys_params.n_j, nq=sys_params.n_q, nv=sys_params.n_v
        )
        uact = CartPoleUnderactuation(sys_params)
    else:
        dyn = CartPoleDynamics(sys_params)

print(sqp_params)
print(prob_params)
print(sys_params)

# Create problem
print(f"Solving..")
prob = Problem(prob_params)
initial_guess = SqpSolution(
    x=torch.zeros((prob.batch_size, prob.horizon, prob.n_x)),
    u=torch.zeros((prob.batch_size, prob.horizon - 1, prob.n_u)),
    mu=torch.zeros((prob.batch_size, prob.horizon, prob.n_x)),
    nu=torch.zeros((prob.batch_size, prob.horizon - 1, prob.n_h)),
    ksi=[None] * prob.horizon,
)

# Costs
Q = prob_params.q_w * torch.eye(dyn.nx).repeat(prob_params.batch_size, 1, 1)
R = prob_params.r_w * torch.eye(dyn.nu).repeat(prob_params.batch_size, 1, 1)
Qf = prob_params.qf_w * torch.eye(dyn.nx).repeat(prob_params.batch_size, 1, 1)

# Set stage costs an initial guess
for k in range(prob.horizon - 1):
    initial_guess.x[:, k] = prob_params.x_init.detach().clone()
    prob.costs.append([LqrCost(Q=Q, R=R)])
    prob.constraints[k] = [
        StateBounds(
            prob.n_x,
            prob.n_u,
            prob_params.x_lb,
            prob_params.x_ub,
        ),
        ControlBounds(
            prob.n_x,
            prob.n_u,
            prob_params.u_lb,
            prob_params.u_ub,
        ),
    ]
# Set terminal cost
initial_guess.x[:, -1] = prob_params.x_des.detach().clone()
prob.costs.append([LqrCost(Q=Qf, x_des=prob_params.x_des.detach().clone())])

# Constraints
if prob_params.inverse_dynamics:
    prob.dynamics = dyn
    prob.underactuation = uact
else:
    prob.dynamics = dyn

# Solve
solution, log = sqp_solve(prob, sqp_params, initial_guess)

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


def plot_controls(controls_tensor):
    first_batch = controls_tensor[0, :, :].detach().cpu().numpy()

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
    # plt.savefig("state_trajectory.png")
    plt.show()


plot_states(solution.x)
plot_controls(solution.u)

# Animate:
if sys_params.name == "acrobot":
    anim = AcrobotAnimator(
        solution.x,
        sys_params.l1,
        sys_params.l2,
        prob_params.dt,
        prob_params.batch_size,
    )
elif sys_params.name == "cartpole":
    anim = CartPoleAnimator(
        solution.x,
        sys_params.lp,
        prob_params.dt,
        prob_params.batch_size,
    )

anim.animate(step_size=2)
# anim.save(filename="four_batches.mp4", step_size=2)
