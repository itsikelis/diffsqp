import torch
from torch.func import jacrev, vmap

from diffsqp.dynamics.base_dynamics import Dynamics
from diffsqp.utils.math import quat_to_rot, q_left


class FloatingBaseDynamics(Dynamics):
    """
    Kinematic double integrator for floating base systems.
    State x: [pos(3), quat(4), vel(3), omega(3)]
    Input u: [v_dot(3), omega_dot(3)]
    """

    def __init__(self, nx, nu, nq, nv):
        super().__init__(nx=nx, nu=nu, nq=nq, nv=nv)

    def fc(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Continuous time kinematic dynamics: x_dot = fc(x, u)
        """
        # Extract states
        # pos = x[..., 0:3]  # Not needed for the derivative
        q = x[..., 3:7]
        v = x[..., 7:10]
        w = x[..., 10:13]

        # Extract accelerations (treated as controls in inverse dynamics)
        v_dot = u[..., 0:3]
        w_dot = u[..., 3:6]

        # 1. Linear velocity
        p_dot = v

        # 2. Quaternion kinematics (omega to q_dot transformation)
        # Normalize the quaternion to prevent numerical drift
        q = q / torch.norm(q, dim=-1, keepdim=True)

        # Create pure quaternion for angular velocity: [0, wx, wy, wz]
        zero_w = torch.cat([torch.zeros_like(w[..., 0:1]), w], dim=-1)

        # q_dot = 0.5 * q_left(q) @ [0, w]^T
        q_L = q_left(q)
        q_dot = 0.5 * torch.einsum("...ij,...j->...i", q_L, zero_w)

        # 3. & 4. The derivatives of the velocities are simply the inputs (accelerations)
        return torch.cat([p_dot, q_dot, v_dot, w_dot], dim=-1)

    def fcx(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        dfc/dx matrix: n_x x n_x
        """

        def fc_single(x_s, u_s):
            return self.fc(x_s, u_s)

        if x.dim() > 1:
            batch_shape = x.shape[:-1]
            x_flat = x.view(-1, x.shape[-1])
            u_flat = u.view(-1, u.shape[-1])

            jac = vmap(jacrev(fc_single, argnums=0))(x_flat, u_flat)
            return jac.view(*batch_shape, jac.shape[-2], jac.shape[-1])
        else:
            return jacrev(fc_single, argnums=0)(x, u)

    def fcu(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        dfc/du matrix: n_x x n_u
        """

        def fc_single(x_s, u_s):
            return self.fc(x_s, u_s)

        if x.dim() > 1:
            batch_shape = x.shape[:-1]
            x_flat = x.view(-1, x.shape[-1])
            u_flat = u.view(-1, u.shape[-1])

            jac = vmap(jacrev(fc_single, argnums=1))(x_flat, u_flat)
            return jac.view(*batch_shape, jac.shape[-2], jac.shape[-1])
        else:
            return jacrev(fc_single, argnums=1)(x, u)

    def calc_semi_impl_matrix_(self, dt):
        E = torch.zeros((self.nx, self.nx))
        # Position + Quaternion kinematics (7x7)
        E[0 : self.nq, 0 : self.nq] = torch.eye(self.nq) * dt
        # Semi-implicit term for linear translation only (3x3)
        E[0:3, self.nq : self.nq + 3] = torch.eye(3) * dt * dt
        # Velocity dynamics (6x6)
        E[self.nq :, self.nq :] = torch.eye(self.nv) * dt

        return E
