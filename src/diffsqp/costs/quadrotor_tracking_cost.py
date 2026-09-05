import torch
from torch.func import jacrev, hessian, vmap
from typing import Optional
from abc import ABC, abstractmethod

from diffsqp.costs import Cost
from diffsqp.utils.math import q_left


class QuadrotorTrackingCost(Cost):
    def __init__(self, Q_diag: torch.Tensor, x_des: torch.Tensor):
        """
        x_des: Reference state tensor of shape (..., 13)
        Q_diag: 1D Tensor of shape (12,) containing weights for:
                [pos(3), ori_error(3), vel(3), omega(3)]
        """
        super().__init__()
        self.x_des = x_des if x_des is not None else torch.zeros(Q.shape[:-1])
        self.Q_diag = Q_diag

    def compute_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the 12D error vector between the current state and reference.
        x: [pos(3), quat(4), vel(3), omega(3)] -> 13D
        """
        # Ensure x_des broadcasts correctly if it's a single state
        x_r = self.x_des.to(device=x.device, dtype=x.dtype)

        # 1. Position Error (3D)
        pos_err = x[..., 0:3] - x_r[..., 0:3]

        # 2. Orientation Error (3D)
        q = x[..., 3:7]
        q = q / torch.norm(q, dim=-1, keepdim=True)  # Normalize
        q_r = x_r[..., 3:7]
        q_r = q_r / torch.norm(q_r, dim=-1, keepdim=True)

        # Inverse (conjugate) of reference quaternion: [w, -x, -y, -z]
        q_r_inv = torch.cat([q_r[..., 0:1], -q_r[..., 1:4]], dim=-1)

        # q_err = q_ref^{-1} * q
        q_L = q_left(q_r_inv)
        q_err = torch.einsum("...ij,...j->...i", q_L, q)

        # The vector part of the error quaternion (x, y, z) maps to sin(theta/2) * axis
        # It correctly evaluates to 0 when q == q_ref AND when q == -q_ref
        ori_err = q_err[..., 1:4]

        # 3. Velocity Error (3D)
        vel_err = x[..., 7:10] - x_r[..., 7:10]

        # 4. Angular Velocity Error (3D)
        omega_err = x[..., 10:13] - x_r[..., 10:13]

        # Combine into a 12D error vector
        return torch.cat([pos_err, ori_err, vel_err, omega_err], dim=-1)

    def l(self, x: torch.Tensor, u: torch.Tensor = None) -> torch.Tensor:
        """
        Evaluates 0.5 * e(x)^T * Q * e(x)
        """
        err = self.compute_error(x)
        Q = self.Q_diag.to(device=x.device, dtype=x.dtype)

        cost = 0.5 * torch.sum(Q * (err**2), dim=-1)
        return cost

    # ---------------------------------------------------------
    # State Derivatives (Computed via Autodiff)
    # ---------------------------------------------------------
    def lx(self, x: torch.Tensor, u: torch.Tensor = None) -> torch.Tensor:
        def l_single(x_s):
            return self.l(x_s)

        if x.dim() > 1:
            batch_shape = x.shape[:-1]
            x_flat = x.view(-1, x.shape[-1])
            jac = vmap(jacrev(l_single))(x_flat)
            return jac.view(*batch_shape, jac.shape[-1])
        return jacrev(l_single)(x)

    def lxx(self, x: torch.Tensor, u: torch.Tensor = None) -> torch.Tensor:
        def l_single(x_s):
            return self.l(x_s)

        if x.dim() > 1:
            batch_shape = x.shape[:-1]
            x_flat = x.view(-1, x.shape[-1])
            H = vmap(hessian(l_single))(x_flat)
            return H.view(*batch_shape, H.shape[-2], H.shape[-1])
        return hessian(l_single)(x)

    # ---------------------------------------------------------
    # Control Derivatives (Hardcoded to zero for speed)
    # Tracking cost depends only on state x, so all u-derivatives are 0.
    # ---------------------------------------------------------
    def lu(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(u)

    def luu(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        batch_shape = u.shape[:-1]
        nu = u.shape[-1]
        return torch.zeros((*batch_shape, nu, nu), device=u.device, dtype=u.dtype)

    def lux(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        batch_shape = u.shape[:-1]
        nu, nx = u.shape[-1], x.shape[-1]
        return torch.zeros((*batch_shape, nu, nx), device=u.device, dtype=u.dtype)

    def lxu(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        batch_shape = u.shape[:-1]
        nx, nu = x.shape[-1], u.shape[-1]
        return torch.zeros((*batch_shape, nx, nu), device=u.device, dtype=u.dtype)
