import torch

from diffsqp.problems import Problem
from diffsqp.costs import LqrCost, TerminalCost
from diffsqp.dynamics import PendulumDynamics
from diffsqp.solvers import Lqr
from diffsqp.solvers import Admm
from diffsqp.solvers import Sqp

# torch.set_default_device("cuda")

horizon = 30
dt = 0.05
n_batch = 3
n_state = 2
n_ctrl = 1
x_des = torch.tensor([torch.pi, 0.0]).repeat(n_batch, 1)

prob = Problem(horizon, dt, n_state, n_ctrl)

dyn = PendulumDynamics(m=1.0, l=1.0, b=0.2, grav=9.81)
Q = 1e-5 * torch.eye(n_state).repeat(n_batch, 1, 1)
R = 1e-3 * torch.eye(n_ctrl).repeat(n_batch, 1, 1)
cost = LqrCost(Q, R)

Qf = 1e6 * torch.eye(n_state).repeat(n_batch, 1, 1)
final_cost = TerminalCost(Qf, x_des)

# Set stage cost and constraints
for i in range(horizon - 1):
    prob.states.append(torch.zeros((n_batch, n_state)))
    prob.controls.append(torch.zeros((n_batch, n_ctrl)))
    prob.costs.append(cost)
    prob.stage_dynamics.append(dyn)
# Set terminal cost
# prob.states.append(torch.zeros((n_batch, n_state)))
prob.states.append(x_des)
prob.costs.append(final_cost)

# Create solver object
# solver = Lqr(prob)
# solver.solve()

# solver = Admm(prob)
# solver.step()

solver = Sqp(prob)

solver.solve()

import matplotlib.pyplot as plt


def plot_states(states_list):
    # 1. Stack the list of tensors into one tensor: (horizon, n_batch, n_x)
    states_tensor = torch.stack(states_list)

    # 2. Extract the first batch (index 0) and convert to numpy
    # Shape becomes: (horizon, n_x)
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


plot_states(prob.states)
