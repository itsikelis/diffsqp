import torch
from torch.func import jacrev, vmap

from diffsqp.constraints import UnderactuationConstraint
from diffsqp.dynamics import QuadrotorParameters
from diffsqp.utils.math import quat_to_rot


class QuadrotorUnderactuation(UnderactuationConstraint):
    def __init__(self, params):
        # n_h = 2 (underactuated constraints: local x and y forces must be 0)
        # n_x = 13 (states: pos(3), quat(4), vel(3), omega(3))
        # n_u = 6 (accelerations: v_dot(3), omega_dot(3))
        super().__init__(n_h=2, n_x=13, n_u=6)
        self.p = params

    def h(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Computes the underactuation residual for the quadrotor.
        x: States [pos(3), quat(4), vel(3), omega(3)]
        u: Accelerations [v_dot(3), omega_dot(3)]
        """
        m = self.p.mass
        g = self.p.grav

        # Extract orientation (quaternion) and linear accelerations
        q = x[..., 3:7]
        v_dot = u[..., 0:3]

        # Normalize quaternion to prevent numerical drift
        q = q / torch.norm(q, dim=-1, keepdim=True)
        R = quat_to_rot(q)  # Shape: (..., 3, 3)

        # 1. Calculate the total required force in the World frame to achieve v_dot
        # F_world = m * v_dot + m * g * e3
        F_world_xy = m * v_dot[..., 0:2]
        F_world_z = m * v_dot[..., 2:3] + (m * g)
        F_world = torch.cat([F_world_xy, F_world_z], dim=-1)

        # 2. Rotate the required world force into the Quadrotor's Body frame
        # F_body = R^T * F_world
        R_T = R.transpose(-1, -2)
        F_body = (R_T @ F_world.unsqueeze(-1)).squeeze(-1)

        # 3. Underactuation Constraint
        # The quadrotor can only generate force (thrust) along its local Z-axis.
        # Thus, the required forces along the local X and Y axes must be 0.
        res = F_body[..., 0:2]

        return res

    def hx(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Jacobian of the underactuation constraint wrt state: dh/dx
        Returns shape: (..., n_h, n_x) -> (..., 2, 13)
        """

        def h_single(x_s, u_s):
            return self.h(x_s, u_s)

        if x.dim() > 1:
            batch_shape = x.shape[:-1]
            x_flat = x.view(-1, x.shape[-1])
            u_flat = u.view(-1, u.shape[-1])

            # argnums=0 differentiates with respect to the first argument (x_s)
            jac = vmap(jacrev(h_single, argnums=0))(x_flat, u_flat)
            return jac.view(*batch_shape, jac.shape[-2], jac.shape[-1])
        else:
            return jacrev(h_single, argnums=0)(x, u)

    def hu(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Jacobian of the underactuation constraint wrt accelerations: dh/du
        Returns shape: (..., n_h, n_u) -> (..., 2, 6)
        """

        def h_single(x_s, u_s):
            return self.h(x_s, u_s)

        if x.dim() > 1:
            batch_shape = x.shape[:-1]
            x_flat = x.view(-1, x.shape[-1])
            u_flat = u.view(-1, u.shape[-1])

            # argnums=1 differentiates with respect to the second argument (u_s)
            jac = vmap(jacrev(h_single, argnums=1))(x_flat, u_flat)
            return jac.view(*batch_shape, jac.shape[-2], jac.shape[-1])
        else:
            return jacrev(h_single, argnums=1)(x, u)
