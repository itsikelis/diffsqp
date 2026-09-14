import os
import tempfile

import torch
from torch.autograd import Variable
from torch import nn

import numpy as np
import matplotlib.pyplot as plt
from mpc import mpc
from mpc.mpc import QuadCost, GradMethods

from tqdm import tqdm


class DoubleIntegratorDx(nn.Module):
    def __init__(self, params=None):
        super().__init__()

        # 2D Double Integrator (x, y)
        self.nq = 2
        self.nv = 2
        self.n_state = self.nq + self.nv  # [x, y, dx, dy]
        self.n_ctrl = 2  # [ddx, ddy]

        self.dt = 0.05

        # Actuation limits
        self.max_force = 10.0
        self.lower = -self.max_force
        self.upper = self.max_force

        # Goal state: Rest at the origin
        self.goal_state = torch.tensor([0.0, 0.0, 0.0, 0.0])

        # Penalties/Weights
        self.goal_weights = torch.tensor([1.0, 1.0, 0.1, 0.1])
        self.ctrl_penalty = 0.01

    def forward(self, state, u):
        squeeze = state.ndimension() == 1
        if squeeze:
            state = state.unsqueeze(0)
            u = u.unsqueeze(0)

        # Clamp control inputs
        u = torch.clamp(u, self.lower, self.upper)

        # Continuous time dynamics: x_dot = fc(x_k, u_k)
        q_dot = state[..., self.nq :]
        q_ddot = u[..., :]
        x_dot = torch.cat([q_dot, q_ddot], dim=-1)

        # Discrete-time dynamics: x_k+1 = x_k + dt * x_dot
        next_state = state + self.dt * x_dot

        if squeeze:
            next_state = next_state.squeeze(0)

        return next_state

    def get_frame(self, state, ax=None):
        """Basic 2D projection/plot for the Double Integrator."""
        state = state.detach().cpu().numpy()
        assert len(state) == 4

        x, y = state[0], state[1]

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig = ax.get_figure()

        ax.scatter(x, y, color="b", s=100, label="Point Mass")

        # Setting static limits to keep the video bounding box consistent
        ax.set_xlim((-5, 5))
        ax.set_ylim((-5, 5))
        ax.set_aspect("equal")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")

        return fig, ax

    def get_true_obj(self):
        q = torch.cat((self.goal_weights, self.ctrl_penalty * torch.ones(self.n_ctrl)))
        px = -torch.sqrt(self.goal_weights) * self.goal_state
        p = torch.cat((px, torch.zeros(self.n_ctrl)))
        return Variable(q), Variable(p)


# Initialize Dynamics
dx = DoubleIntegratorDx()
n_batch, T, mpc_T = 8, 100, 25


def uniform(shape, low, high):
    r = high - low
    return torch.rand(shape) * r + low


# Initialize 4D State: [x, y, dx, dy]
pos = uniform((n_batch, 2), -2.0, 2.0)
vel = uniform((n_batch, 2), -0.5, 0.5)
xinit = torch.cat((pos, vel), dim=1)

x = xinit
u_init = None

q, p = dx.get_true_obj()
Q = torch.diag(q).unsqueeze(0).unsqueeze(0).repeat(mpc_T, n_batch, 1, 1)
p = p.unsqueeze(0).repeat(mpc_T, n_batch, 1)

t_dir = tempfile.mkdtemp()
print("Tmp dir: {}".format(t_dir))

nominal_states, nominal_actions, nominal_objs = mpc.MPC(
    dx.n_state,
    dx.n_ctrl,
    mpc_T,
    u_init=u_init,
    u_lower=dx.lower,
    u_upper=dx.upper,
    lqr_iter=50,
    verbose=0,
    exit_unconverged=False,
    detach_unconverged=False,
    linesearch_decay=getattr(dx, "linesearch_decay", 0.5),
    max_linesearch_iter=getattr(dx, "max_linesearch_iter", 10),
    grad_method=GradMethods.AUTO_DIFF,
    eps=1e-4,
)(x, QuadCost(Q, p), dx)
