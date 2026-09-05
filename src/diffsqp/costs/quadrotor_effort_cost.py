from typing import Optional

import torch
from torch.func import jacrev, hessian, vmap

from diffsqp.costs import Cost
from diffsqp.utils.math import quat_to_rot


class QuadrotorEffortCost(Cost):
    def __init__(self, params, R_diag: torch.Tensor):
        """
        params: QuadrotorParameters instance.
        R_diag: 1D Tensor of shape (4,) containing weights for [thrust, tau_x, tau_y, tau_z].
        """
        super().__init__()
        self.p = params
        self.R_diag = R_diag

    def compute_physical_controls(
        self, x: torch.Tensor, u_accel: torch.Tensor
    ) -> torch.Tensor:
        """
        Recovers the physical [thrust, torques] from the kinematic states and accelerations.
        x: [pos(3), quat(4), vel(3), omega(3)]
        u: [v_dot(3), omega_dot(3)]
        """
        device, dtype = x.device, x.dtype
        m = self.p.mass
        g = self.p.grav
        inertia = self.p.inertia.to(device=device, dtype=dtype)

        q = x[..., 3:7]
        w = x[..., 10:13]
        v_dot = u_accel[..., 0:3]
        w_dot = u_accel[..., 3:6]

        # Normalize quaternion
        q = q / torch.norm(q, dim=-1, keepdim=True)
        R = quat_to_rot(q)

        # 1. Recover Thrust (Local Z component of total required force)
        F_world_xy = m * v_dot[..., 0:2]
        F_world_z = m * v_dot[..., 2:3] + (m * g)
        F_world = torch.cat([F_world_xy, F_world_z], dim=-1)

        R_T = R.transpose(-1, -2)
        F_body = (R_T @ F_world.unsqueeze(-1)).squeeze(-1)

        thrust = F_body[..., 2:3]

        # 2. Recover Torques (tau = I * w_dot + w x (I * w))
        Iw = (inertia @ w.unsqueeze(-1)).squeeze(-1)
        tau = (inertia @ w_dot.unsqueeze(-1)).squeeze(-1) + torch.cross(w, Iw, dim=-1)

        return torch.cat([thrust, tau], dim=-1)

    def l(self, x: torch.Tensor, u: torch.Tensor = None) -> torch.Tensor:
        """
        Evaluates 0.5 * physical_u^T * R * physical_u
        """
        if u is None:
            return torch.zeros((*x.shape[:-1],), device=x.device, dtype=x.dtype)

        u_phys = self.compute_physical_controls(x, u)
        R = self.R_diag.to(device=x.device, dtype=x.dtype)

        # Element-wise multiplication for diagonal R, then sum
        cost = 0.5 * torch.sum(R * (u_phys**2), dim=-1)
        return cost

    # ---------------------------------------------------------
    # Autodiff Helpers
    # ---------------------------------------------------------

    def _batched_jacobian(self, func, argnums, x, u):
        def f_single(x_s, u_s):
            return func(x_s, u_s)

        if x.dim() > 1:
            batch_shape = x.shape[:-1]
            x_flat, u_flat = x.view(-1, x.shape[-1]), u.view(-1, u.shape[-1])
            jac = vmap(jacrev(f_single, argnums=argnums))(x_flat, u_flat)
            return jac.view(*batch_shape, *jac.shape[1:])
        return jacrev(f_single, argnums=argnums)(x, u)

    def _batched_hessian(self, func, argnums, x, u):
        def f_single(x_s, u_s):
            return func(x_s, u_s)

        if x.dim() > 1:
            batch_shape = x.shape[:-1]
            x_flat, u_flat = x.view(-1, x.shape[-1]), u.view(-1, u.shape[-1])
            H = vmap(hessian(f_single, argnums=argnums))(x_flat, u_flat)
            return H.view(*batch_shape, *H.shape[1:])
        return hessian(f_single, argnums=argnums)(x, u)

    def _batched_mixed_hessian(self, func, argnums_inner, argnums_outer, x, u):
        """
        Computes mixed derivatives, e.g., d(dl/dx)/du.
        """

        def f_single(x_s, u_s):
            # Compute first derivative, then trace again for the second
            inner_grad_fn = jacrev(func, argnums=argnums_inner)
            return jacrev(inner_grad_fn, argnums=argnums_outer)(x_s, u_s)

        if x.dim() > 1:
            batch_shape = x.shape[:-1]
            x_flat, u_flat = x.view(-1, x.shape[-1]), u.view(-1, u.shape[-1])
            H = vmap(f_single)(x_flat, u_flat)
            return H.view(*batch_shape, *H.shape[1:])
        return f_single(x, u)

    # ---------------------------------------------------------
    # API Methods
    # ---------------------------------------------------------

    def lx(self, x: torch.Tensor, u: torch.Tensor = None) -> torch.Tensor:
        if u is None:
            return torch.zeros_like(x)
        return self._batched_jacobian(self.l, 0, x, u)

    def lu(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self._batched_jacobian(self.l, 1, x, u)

    def lxx(self, x: torch.Tensor, u: torch.Tensor = None) -> torch.Tensor:
        if u is None:
            return torch.zeros(
                (*x.shape[:-1], x.shape[-1], x.shape[-1]),
                device=x.device,
                dtype=x.dtype,
            )
        return self._batched_hessian(self.l, 0, x, u)

    def luu(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self._batched_hessian(self.l, 1, x, u)

    def lux(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # Derivative of lu (arg 1) with respect to x (arg 0)
        return self._batched_mixed_hessian(
            self.l, argnums_inner=1, argnums_outer=0, x=x, u=u
        )

    def lxu(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # Derivative of lx (arg 0) with respect to u (arg 1)
        return self._batched_mixed_hessian(
            self.l, argnums_inner=0, argnums_outer=1, x=x, u=u
        )
