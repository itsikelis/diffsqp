import time
import torch
import numpy as np

from diffsqp.problems import Problem
from diffsqp.costs import LqrCost
from diffsqp.dynamics import Dynamics
from diffsqp.constraints import AcrobotUnderactuation
from diffsqp.utils.animate import AcrobotAnimator
from diffsqp.solvers import Lqr
from diffsqp.solvers import Sqp, SqpParams

# torch.set_default_dtype(torch.double)
# torch.set_default_device("cuda")

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

l1 = 0.5
l2 = 0.5
dt = 0.01
tf = 1.0
horizon = int(tf / dt)
n_B = 3
nx = 4
nu = 2

dyn = Dynamics(nx=nx, nu=nu, nq=2, nv=2)
uact = AcrobotUnderactuation(
    m1=1.0,
    m2=1.0,
    l1=l1,
    l2=l2,
    lc1=0.5,
    lc2=0.5,
    grav=9.81,
    I2=1 / (3.0 * 1.0 * 0.5**2),
    I1=1 / (3.0 * 1.0 * 0.5**2),
)

# x_init = torch.tensor([torch.pi, 0.0, 0.0, 0.0]).repeat(n_B, 1)
x_init = torch.tensor([0.0, 0.0, 0.0, 0.0]).repeat(n_B, 1)
# x_init[:, 0:2] += 0.2 * torch.randn((n_B, 2))
x_des = torch.tensor([torch.pi, 0.0, 0.0, 0.0]).repeat(n_B, 1)

prob = Problem(horizon, dt, n_B, nx, nu)

q_w = torch.tensor([1e-6, 1e-6, 1e-6, 1e-6])
r_w = torch.tensor([1e-1])
qf_w = torch.tensor([4e8, 4e8, 1e5, 1e5])

Q = q_w * torch.eye(nx).repeat(n_B, 1, 1)
R = r_w * torch.eye(nu).repeat(n_B, 1, 1)
Qf = qf_w * torch.eye(nx).repeat(n_B, 1, 1)

# Set stage cost and constraints
for i in range(horizon - 1):
    prob.states[i] = x_init.clone()
    prob.costs.append([LqrCost(Q=Q, R=R)])
    prob.dynamics.append(dyn)
    prob.constraints[i] = [uact]

# Set terminal cost
prob.states[-1] = x_des.clone()
prob.costs.append([LqrCost(Q=Qf, x_des=x_des.clone())])

# Create solver object
# qp_solver = Lqr(prob)
sqp_params = SqpParams(
    qp_solver="lqr", ls_technique="merit", n_B=n_B, max_iter=500, eps=1e-4
)
solver = Sqp(prob, sqp_params)

start = time.time()
try:
    info = solver.solve()
except KeyboardInterrupt:
    print("Keyboard  Interrupt")
end = time.time()

print("Time elapsed: ", end - start, " s.")

import matplotlib.pyplot as plt


def plot_states(states_list):
    # 1. Stack the list of tensors into one tensor: (horizon, n_B, n_x)
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

anim = AcrobotAnimator(np.array(prob.states), l1, l2, dt, n_B)
anim.animate(step_size=2)
