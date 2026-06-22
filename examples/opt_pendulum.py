import torch

from diffsqp.problems import Problem
from diffsqp.costs import LqrCost
from diffsqp.dynamics import PendulumDynamics
from diffsqp.solvers import Lqr
from diffsqp.solvers import Sqp, SqpParameters

# torch.set_default_device("cuda")

horizon = 30
dt = 0.05
nB = 3
nx = 2
nu = 1
x_des = torch.tensor([torch.pi, 0.0]).repeat(nB, 1)

prob = Problem(horizon, dt, nB, nx, nu)

dyn = PendulumDynamics(m=1.0, l=1.0, b=0.2, grav=9.81)
Q = 1e-5 * torch.eye(nx).repeat(nB, 1, 1)
R = 1e-3 * torch.eye(nu).repeat(nB, 1, 1)

Qf = 1e6 * torch.eye(nx).repeat(nB, 1, 1)

# Set stage cost and constraints
for i in range(horizon - 1):
    prob.states[i] = torch.zeros((nB, nx))
    prob.costs.append([LqrCost(Q, R)])
    prob.dynamics.append(dyn)

# Set terminal cost
# prob.states.append(torch.zeros((nB, nx)))
prob.states[-1] = x_des.clone()
prob.costs.append([LqrCost(Q=Qf, x_des=x_des)])

# Create solver object
# solver = Lqr(prob)
# solver.solve()

# solver = Admm(prob)
# solver.step()

sqp_params = SqpParameters(qp_solver="lqr", n_B=n_B, max_iter=500, eps=1e-4)
solver = Sqp(prob, sqp_params)

solver.solve()

import matplotlib.pyplot as plt


def plot_states(states_list):
    # 1. Stack the list of tensors into one tensor: (horizon, nB, n_x)
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
