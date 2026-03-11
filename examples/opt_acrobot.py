import time
import torch

import numpy as np

from diffsqp.problems import Problem
from diffsqp.costs import LqrCost, TerminalCost
from diffsqp.dynamics import AcrobotDynamics, AcrobotInverseDynamics
from diffsqp.utils.animate import AcrobotAnimator
from diffsqp.solvers import Lqr
from diffsqp.solvers import Admm
from diffsqp.solvers import Sqp

# torch.set_default_dtype(torch.double)
# torch.set_default_device("cuda")

dyn = AcrobotInverseDynamics(
    m1=1.0,
    m2=1.0,
    l1=0.5,
    l2=0.5,
    lc1=0.5,
    lc2=0.5,
    grav=9.81,
    I2=1 / (3.0 * 1.0 * 0.5**2),
    I1=1 / (3.0 * 1.0 * 0.5**2),
)

## Shivesh acrobot parametres
# m1=0.10548177618443695,
# m2=0.07619744360415454,
# l1=0.05,
# l2=0.05,
# lc1=0.05,
# lc2=0.03670036749567022,
# grav=9.81,
# I2=0.00023702395072092597,
# I1=0.00046166221821039165,

dt = 0.01
tf = 1.0
horizon = int(tf / dt)
n_batch = 4
n_state = dyn.nx
n_ctrl = dyn.nu

x_init = torch.tensor([torch.pi, 0.0, 0.0, 0.0]).repeat(n_batch, 1)
x_init[:, 0:2] += 0.2 * torch.randn((n_batch, 2))
x_des = torch.tensor([torch.pi, 0.0, 0.0, 0.0]).repeat(n_batch, 1)

prob = Problem(horizon, dt, n_state, n_ctrl)

q_w = torch.tensor([1e-6, 1e-6, 1e-6, 1e-6])
r_w = torch.tensor([1e-1])
qf_w = torch.tensor([4e8, 4e8, 1e5, 1e5])

Q = q_w * torch.eye(n_state).repeat(n_batch, 1, 1)
R = r_w * torch.eye(n_ctrl).repeat(n_batch, 1, 1)
Qf = qf_w * torch.eye(n_state).repeat(n_batch, 1, 1)

# Set stage cost and constraints
for i in range(horizon - 1):
    if i == 0:
        prob.states.append(x_init.clone())
    else:
        prob.states.append(x_des.clone())
    prob.controls.append(torch.zeros((n_batch, n_ctrl)))
    prob.costs.append(LqrCost(Q, R, x_des.clone()))
    prob.stage_dynamics.append(dyn)
# Set terminal cost prob.states.append(torch.zeros((n_batch, n_state)))
prob.states.append(x_des.clone())
prob.costs.append(TerminalCost(Qf, x_des.clone()))

# Create solver object
qp_solver = Lqr(prob)
solver = Sqp(prob, qp_solver, max_iter=1000)

start = time.time()

try:
    solver.solve()
except KeyboardInterrupt:
    print("Keyboard  Interrupt")

end = time.time()
print("Time elapsed: ", end - start, " s.")

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

anim = AcrobotAnimator(np.array(prob.states), dyn.l1, dyn.l2, dt, n_batch)
anim.animate(step_size=2)

print(solver.terminated)
