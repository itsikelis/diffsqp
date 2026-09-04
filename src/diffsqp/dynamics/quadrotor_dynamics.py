import torch
from typing import Any, Dict
from diffsqp.dynamics.base_dynamics import Dynamics
from torch.func import jacrev, vmap

from diffsqp.utils.math import quat_to_rot, q_left


# Batched Skew symmetric map (lie algebra of SO(3), solves a x b = S(a)b)
def S(u: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(u[..., 0])
    row0 = torch.stack([z, -u[..., 2], u[..., 1]], dim=-1)
    row1 = torch.stack([u[..., 2], z, -u[..., 0]], dim=-1)
    row2 = torch.stack([-u[..., 1], u[..., 0], z], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


class QuadrotorParameters:
    def __init__(self, **kwargs):
        self.name: str = kwargs.get("name", "Quadrotor")

        self.n_x: int = kwargs.get("n_x", 13)
        self.n_q: int = kwargs.get("n_q", 7)  # pos(3) + quat(4)
        self.n_v: int = kwargs.get("n_v", 6)  # vel(3) + omega(3)
        self.n_j: int = kwargs.get("n_j", 0)
        self.n_u: int = kwargs.get("n_u", 4)

        self.mass: float = kwargs.get("mass", 0.1)

        # Expects a flat list of 9 elements or a 3x3 array/tensor
        inertia_val = kwargs.get(
            "inertia", [0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1]
        )
        self.inertia: torch.Tensor = torch.tensor(
            inertia_val, dtype=torch.float32
        ).view(3, 3)
        self.inertia_inv: torch.Tensor = torch.linalg.inv(self.inertia)

        self.grav: float = kwargs.get("grav", 9.81)

    def __str__(self) -> str:
        return (
            f"=== Quadrotor Parameters ===\n"
            f"  Name            : {self.name}\n"
            f"  State Dim (nx)  : {self.n_x}\n"
            f"  Pos Dim (nq)    : {self.n_q}\n"
            f"  Vel Dim (nv)    : {self.n_v}\n"
            f"  Control Dim (nu): {self.n_u}\n"
            f"------------------\n"
            f"  Mass            : {self.mass:.3f}\n"
            f"  Gravity (grav)  : {self.grav:.3f}\n"
            f"==========================="
        )


class QuadrotorDynamics(Dynamics):
    """Batched Quadrotor dynamics class using right-handed quaternion (not JPL convention)."""

    def __init__(self, params: QuadrotorParameters):
        super().__init__(nx=params.n_x, nu=params.n_u, nq=params.n_q, nv=params.n_v)
        self.p = params

    def fc(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Continuous time dynamics: x_dot = fc(x_k, u_k)
        """
        device, dtype = x.device, x.dtype
        inertia = self.p.inertia.to(device=device, dtype=dtype)
        inertia_inv = self.p.inertia_inv.to(device=device, dtype=dtype)

        m = self.p.mass
        g = self.p.grav

        q = x[..., 3:7]
        v = x[..., 7:10]
        w = x[..., 10:13]
        f = u[..., 0:1]
        tau = u[..., 1:4]

        # Normalize the quaternion
        q = q / torch.norm(q, dim=-1, keepdim=True)

        e3 = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        R = quat_to_rot(q)

        # Helper for batched matrix-vector multiplication
        def bmv(mat, vec):
            return (mat @ vec.unsqueeze(-1)).squeeze(-1)

        p_dot = v
        v_dot = -g * e3 + (f / m) * bmv(R, e3)

        zero_w = torch.cat([torch.zeros_like(w[..., 0:1]), w], dim=-1)
        q_dot = 0.5 * bmv(q_left(q), zero_w)

        Jw = bmv(inertia, w)
        w_dot = bmv(inertia_inv, bmv(S(Jw), w) + tau)

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
