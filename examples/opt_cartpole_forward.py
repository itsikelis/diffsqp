import time
import torch

from diffsqp.problems import Problem
from diffsqp.costs import LqrCost, TerminalCost
from diffsqp.dynamics import (
    CartPoleDynamics,
    CartPoleInverseDynamics,
    CartPoleInverseDynamicsConstrained,
)
from diffsqp.solvers import Lqr
from diffsqp.solvers import Admm
from diffsqp.solvers import Sqp

from diffsqp.utils.animate import CartPoleAnimator

# torch.set_default_dtype(torch.double)
# torch.set_default_device("cuda")

dyn = CartPoleDynamics(mc=0.5, mp=0.3, lp=0.2, grav=9.81)

dt = 0.01
tf = 1.0
horizon = int(tf / dt)
nB = 2
nx = dyn.nx
nu = dyn.nu

# x_init = torch.tensor(
#     [
#         [0.0, 0.0, 0.0, 0.0],
#         [0.1, torch.pi, 0.0, 0.0],
#         [-0.1, torch.pi, 0.0, 0.0],
#         [-4.6296e-02, 2.8597e00, 2.8562e-01, 2.3995e00],
#     ]
# )
x_des = torch.tensor([0.0, torch.pi, 0.0, 0.0]).repeat(nB, 1)
x_init = torch.tensor([0.0, 0.0, 0.0, 0.0]).repeat(nB, 1)
x_init[:, 0:2] += 0.1 * torch.randn((nB, 2))

prob = Problem(horizon, dt, nx, nu)
q_w = torch.tensor([1e-6, 1e-6, 1e-6, 1e-6])
r_w = torch.tensor([1e-3])
qf_w = torch.tensor([1e5, 1e5, 1e5, 1e5])

Q = q_w * torch.eye(nx).repeat(nB, 1, 1)
R = r_w * torch.eye(nu).repeat(nB, 1, 1)
Qf = qf_w * torch.eye(nx).repeat(nB, 1, 1)

# Set stage cost and constraints
for i in range(horizon - 1):
    prob.states.append(x_init.clone())
    prob.controls.append(torch.zeros((nB, nu)))
    prob.costs.append(LqrCost(Q, R))
    prob.stage_dynamics.append(dyn)
    # if i == int(horizon / 2):
    #     prob.stage_dynamics.append(dyn_c)
    # else:
    #     prob.stage_dynamics.append(dyn)

# Set terminal cost
# prob.states.append(torch.zeros((nB, nx)))
prob.states.append(x_des.clone())
prob.costs.append(TerminalCost(Qf, x_des.clone()))

# Create solver object
qp_solver = Lqr(prob)
solver = Sqp(prob, qp_solver)

start = time.time()

info = solver.solve()
print("max_c_viol: ", info["max_uact_viol"])
print("u_k-1: ", prob.controls[int(horizon / 2) - 1][0])
print("u-k: ", prob.controls[int(horizon / 2)][0])
print("u-k+1: ", prob.controls[int(horizon / 2) + 1][0])

end = time.time()
print("Time elapsed: ", end - start, " s.")

import matplotlib.pyplot as plt

# def plot_states(states_list):
#     # 1. Stack the list of tensors into one tensor: (horizon, nB, n_x)
#     states_tensor = torch.stack(states_list)
#
#     # 2. Extract the first batch (index 0) and convert to numpy
#     # Shape becomes: (horizon, n_x)
#     first_batch = states_tensor[:, 0, :].detach().cpu().numpy()
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
#     plt.title("State Trajectory (First Batch)")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     # plt.savefig("state_trajectory.png")
#     plt.show()
#
#
# plot_states(prob.states)

print(solver.terminated)

import numpy as np

anim = CartPoleAnimator(np.array(prob.states), dyn.lp, dt, nB)
anim.animate(step_size=2)
# anim.save(filename="four_batches.mp4", step_size=2)
