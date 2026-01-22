import torch

from diffsqp.problems import Problem
from diffsqp.costs import LqrCost
from diffsqp.dynamics import PendulumDynamics
from diffsqp.solvers import AdmmSolver
from diffsqp.solvers import LqrSolver

horizon = 10
dt = 0.05
n_batch = 3
n_state = 2
n_ctrl = 1
prob = Problem(horizon, dt, n_state, n_ctrl)

dyn = PendulumDynamics(m=1.0, l=1.0, b=0.2, grav=9.81)
Q = 1e-3 * torch.eye(n_state).repeat(n_batch, 1, 1)
R = 0.5 * torch.eye(n_ctrl).repeat(n_batch, 1, 1)
cost = LqrCost(Q, R)

Qf = 1e3 * torch.eye(n_state).repeat(n_batch, 1, 1)
Rf = 0.0 * torch.eye(n_ctrl).repeat(n_batch, 1, 1)
final_cost = LqrCost(Qf, Rf)

# Set stage cost and constraints
for i in range(horizon - 1):
    prob.costs.append(cost)
    prob.stage_dynamics.append(dyn)
# Set terminal cost
prob.costs.append(final_cost)

# Set initial guess
prob.variables = torch.ones((n_batch, n_state * horizon + (horizon - 1) * n_ctrl))

# Create solver object
# solver = LqrSolver(prob)
# solver.solve()

solver = AdmmSolver(prob)
solver.step()

# print(solver.V[0])
# print(solver.u[0])
# print(solver.h)
# print(solver.delta_x[-1])
# print(solver.delta_u[-1])
# print(solver.lagrange_mult[-1])
#
# print(prob.state(2))
# prob.set_state(2, torch.ones((n_batch, n_state)))
# print(prob.state(2))
