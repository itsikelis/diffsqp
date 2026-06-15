import os
import time
import argparse
import torch
import yaml

import numpy as np

from diffsqp.problems import Problem, ProblemParams
from diffsqp.costs import LqrCost
from diffsqp.solvers import Lqr
from diffsqp.solvers import Sqp, SqpParams
from diffsqp.dynamics import Dynamics, AcrobotDynamics, CartPoleDynamics
from diffsqp.dynamics import AcrobotParameters, CartPoleParameters
from diffsqp.constraints import AcrobotUnderactuation, CartPoleUnderactuation
from diffsqp.utils.animate import AcrobotAnimator, CartPoleAnimator


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
sqp_params = SqpParams(**cfg["solver"])
print(sqp_params)
prob_params = ProblemParams(**cfg["problem"])

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

# Create problem
print(f"Solving..")
prob = Problem(prob_params)

# Costs
Q = prob_params.q_w * torch.eye(dyn.nx).repeat(prob_params.n_batch, 1, 1)
R = prob_params.r_w * torch.eye(dyn.nu).repeat(prob_params.n_batch, 1, 1)
Qf = prob_params.qf_w * torch.eye(dyn.nx).repeat(prob_params.n_batch, 1, 1)
# Set stage initial guess and costs
for i in range(prob.horizon - 1):
    prob.states[i] = prob_params.x_init.clone()
    prob.costs.append([LqrCost(Q=Q, R=R)])
# Set terminal cost
prob.states[-1] = prob_params.x_init.clone()
prob.costs.append([LqrCost(Q=Qf, x_des=prob_params.x_des.clone())])

# Constraints
if prob_params.inverse_dynamics:
    prob.dynamics = dyn
    prob.underactuation = uact
else:
    prob.dynamics = dyn


# Create solver and solve
solver = Sqp(prob, sqp_params)

start = time.time()
try:
    log = solver.solve()
except KeyboardInterrupt:
    print("Keyboard  Interrupt")
end = time.time()

print(log)

print("Time elapsed: ", end - start, " s.")


import matplotlib.pyplot as plt


def plot_states(states_list):
    states_tensor = torch.stack(states_list)
    first_batch = states_tensor[:, 0, :].detach().cpu().numpy()

    horizon, n_x = first_batch.shape
    time = range(horizon)

    # 3. Plot each dimension of the state
    for i in range(n_x):
        plt.plot(time, first_batch[:, i], label=f"State $x_{{{i}}}$")

    plt.xlabel("Time Step $k$")
    plt.ylabel("Value")
    plt.title("State Trajectory (First Batch)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_controls(controls_list):
    states_tensor = torch.stack(controls_list)
    first_batch = states_tensor[:, 0, :].detach().cpu().numpy()

    horizon, n_x = first_batch.shape
    time = range(horizon)

    # 3. Plot each dimension of the state
    for i in range(n_x):
        plt.plot(time, first_batch[:, i], label=f"State $x_{{{i}}}$")

    plt.xlabel("Time Step $k$")
    plt.ylabel("Value")
    plt.title("State Trajectory (First Batch)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig("state_trajectory.png")
    plt.show()


# plot_states(prob.states)
# plot_controls(prob.controls)

# Animate:
if sys_params.name == "acrobot":
    anim = AcrobotAnimator(
        np.array(prob.states),
        sys_params.l1,
        sys_params.l2,
        prob_params.dt,
        prob_params.n_batch,
    )
elif sys_params.name == "cartpole":
    anim = CartPoleAnimator(
        np.array(prob.states), sys_params.lp, prob_params.dt, prob_params.n_batch
    )

anim.animate(step_size=2)
# anim.save(filename="four_batches.mp4", step_size=2)
