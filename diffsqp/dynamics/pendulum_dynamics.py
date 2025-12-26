import torch

from diffsqp.dynamics import Dynamics


class PendulumDynamics(Dynamics):
    def __init__(self, m: float, l: float, b: float, grav: float = 9.81):
        self.nq = 1
        self.nv = 1
        self.nu = 1
        self.m = m
        self.l = l
        self.b = b
        self.grav = grav

        # Pre-calculate constant factor
        self.inertia_inv = 1.0 / (self.m * self.l**2)

    def f(self, x, u):
        # x_dot:
        # [ theta_dot  ]
        # [ theta_ddot ]
        theta = x[:, 0:1]
        dtheta = x[:, 1:2]

        ddtheta = (
            -(self.grav / self.l) * torch.sin(theta)
            - (self.b * self.inertia_inv) * dtheta
            + (self.inertia_inv) * u
        )

        return torch.cat([dtheta, ddtheta], dim=1)

    def fx(self, x, u):
        # df/dx matrix:
        # [ 0.0,               1.0     ]
        # [ -g/l * cos(theta), -b/ml^2 ]
        batch_size = x.shape[0]

        # Initialize Jacobian tensor (Batch, State, State)
        A = torch.zeros((batch_size, 2, 2), device=x.device)
        A[:, 0, 1] = 1.0
        A[:, 1, 0] = -(self.grav / self.l) * torch.cos(x[:, 0])
        A[:, 1, 1] = -self.b * self.inertia_inv
        return A

    def fu(self, x, u):
        batch_size = x.shape[0]
        # df/du matrix:
        # [ 0.0    ]
        # [ 1/ml^2 ]

        B = torch.zeros((batch_size, 2, 1), device=x.device)
        B[:, 1, 0] = self.inertia_inv
        return B
