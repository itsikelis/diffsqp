"""Batched Unitree Go2 standing with inverse dynamics: base rows in the Riccati.

The state is x = [q, v] with the gdyn (Pinocchio) free-flyer layout,
q = [base position, base quaternion (x y z w), 12 joint angles] and
v = [base linear and angular velocity (base frame), 12 joint rates]. The
controls are u = [a, f]: the generalized accelerations and the world-frame
forces of the four feet. `Go2Dynamics` integrates them with a semi-implicit
Euler step on the configuration manifold (gdyn.integrate, d_integrate).

The physics lives in the inverse dynamics tau = RNEA(q, v, a) - sum_c J_c^T f_c:

- its 6 floating-base rows must vanish (the base is not actuated). Here they are
  the underactuation constraint h(x, u) = 0 (`Go2BaseUnderactuation`, n_h = 6),
  enforced exactly by the equality-constrained Riccati recursion;
- its 12 joint rows are the motor torques, bounded by `JointTorqueBounds` in ADMM.

`examples/go2_inverse_admm.py` is the same problem with all 18 rows in ADMM.

The feet stay in sticking contact (`FootContact`: fixed positions and zero
velocities) inside linearized friction cones (`FrictionCone`). Each problem of
the batch moves the base from the nominal stance to a random pose offset and
comes to rest. The derivatives of J_c^T f_c and J_c v with respect to q come
from the kinematic Hessian of the foot Jacobians (`jacobian_transpose_force_derivative`,
`foot_velocity_derivative`).

    uv run examples/go2_inverse_riccati.py --batch_size 16 --save out/go2_riccati
"""

import argparse
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import torch
import gdyn

from diffsqp.problems import Problem, ProblemParameters
from diffsqp.costs import LqrCost
from diffsqp.solvers import sqp_solve, SqpParameters
from diffsqp.dynamics import Dynamics
from diffsqp.constraints import (
    GenericConstraint,
    UnderactuationConstraint,
    StateBounds,
    ControlBounds,
)
from diffsqp.types import SqpSolution
from diffsqp.utils.load_save import save_solution

URDF_PATH = Path(__file__).resolve().parent.parent / "resources" / "robots" / "go2.urdf"
FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
# Finite "no bound" (like PAD_BOUND), so complementarity products stay finite
UNBOUNDED = 1e20


@dataclass
class Go2Parameters:
    name: str = "go2"
    n_x: int = 37  # [q (19), v (18)]
    n_u: int = 30  # [a (18), foot forces (12)]
    n_q: int = 19
    n_v: int = 18


def skew(w: torch.Tensor) -> torch.Tensor:
    """Skew-symmetric matrix of w: (..., 3) -> (..., 3, 3)."""
    wx, wy, wz = w.unbind(-1)
    zero = torch.zeros_like(wx)
    return torch.stack(
        [
            torch.stack([zero, -wz, wy], dim=-1),
            torch.stack([wz, zero, -wx], dim=-1),
            torch.stack([-wy, wx, zero], dim=-1),
        ],
        dim=-2,
    )


def normalized(q: torch.Tensor) -> torch.Tensor:
    """q with a unit base quaternion: gdyn assumes valid configurations, and
    the QP steps move the quaternion off the unit sphere."""
    quat = q[..., 3:7]
    quat = quat / torch.norm(quat, dim=-1, keepdim=True)
    return torch.cat([q[..., :3], quat, q[..., 7:]], dim=-1)


def tangent_map(q: torch.Tensor) -> torch.Tensor:
    """
    d(tangent) / d(coordinates), (..., nv, nq), of the configuration
    normalized(q). gdyn differentiates along integrate(q, dq), so a gdyn
    derivative G (..., m, nv) becomes G @ tangent_map(q) in the coordinates of x.

    integrate moves the base by right composition with exp6(dq): to first
    order the position moves by R dq_lin and the quaternion by
    quat (x) [dq_ang / 2, 0], so dq_lin = R^T dp and
    dq_ang = 2 Im(conj(quat) (x) dquat) (zero along the quaternion itself).
    """
    norm = torch.norm(q[..., 3:7], dim=-1, keepdim=True)
    quat = q[..., 3:7] / norm
    vec, w = quat[..., :3], quat[..., 3:]
    vec_x = skew(vec)
    eye3 = torch.eye(3, dtype=q.dtype, device=q.device)
    R = eye3 + 2.0 * w.unsqueeze(-1) * vec_x + 2.0 * vec_x @ vec_x

    T = torch.zeros((*q.shape[:-1], 18, 19), dtype=q.dtype, device=q.device)
    T[..., :3, :3] = R.mT
    T[..., 3:6, 3:6] = 2.0 * (w.unsqueeze(-1) * eye3 - vec_x) / norm.unsqueeze(-1)
    T[..., 3:6, 6] = -2.0 * vec / norm
    T[..., 6:, 7:] = torch.eye(12, dtype=q.dtype, device=q.device)
    return T


def coordinate_map(q: torch.Tensor) -> torch.Tensor:
    """
    d(coordinates) / d(tangent), (..., nq, nv), at a normalized q: the
    coordinates of integrate(q, e) to first order in e, so that
    tangent_map(q) @ coordinate_map(q) = I. The position moves by R e_lin and
    the quaternion by quat (x) [e_ang / 2, 0].
    """
    vec, w = q[..., 3:6], q[..., 6:7]
    vec_x = skew(vec)
    eye3 = torch.eye(3, dtype=q.dtype, device=q.device)
    R = eye3 + 2.0 * w.unsqueeze(-1) * vec_x + 2.0 * vec_x @ vec_x

    E = torch.zeros((*q.shape[:-1], 19, 18), dtype=q.dtype, device=q.device)
    E[..., :3, :3] = R
    E[..., 3:6, 3:6] = 0.5 * (w.unsqueeze(-1) * eye3 + vec_x)
    E[..., 6, 3:6] = -0.5 * vec
    E[..., 7:, 6:] = torch.eye(12, dtype=q.dtype, device=q.device)
    return E


def ancestor_mask(nv: int, device) -> torch.Tensor:
    """P[i, j]: velocity j moves the joint of velocity i. Pinocchio orders
    joints parents first, and the six free-flyer directions all move the base.
    Joints of other legs have zero foot Jacobian columns, so the leg order
    does not matter."""
    idx = torch.arange(nv, device=device)
    return (idx[None, :] <= idx[:, None]) | ((idx[:, None] < 6) & (idx[None, :] < 6))


def foot_kinematics(model, q: torch.Tensor):
    """Foot positions (..., 4, 3) and the linear and angular rows L, A
    (..., 4, 3, nv) of the LOCAL_WORLD_ALIGNED foot Jacobians, at a normalized q."""
    J, oMf = gdyn.frame_jacobians(
        model, q, FEET, gdyn.LOCAL_WORLD_ALIGNED, return_placements=True
    )
    return oMf[..., :3, 3], J[..., :3, :], J[..., 3:, :]


def jacobian_transpose_force_derivative(L, A, f):
    """
    d(sum_c L_c^T f_c) / dq in the tangent space, (..., nv, nv).

    Kinematic Hessian of the foot Jacobians: column L_i changes along velocity
    j as a_j x L_i if j moves joint i, and as a_i x L_j otherwise (a = A
    columns). With X[i, j] = f . (a_j x L_i) = (L_i x f) . a_j, the result is
    X where j moves i and X^T elsewhere.
    """
    Lf = torch.linalg.cross(L, f.unsqueeze(-1).expand_as(L), dim=-2)
    X = (Lf.mT @ A).sum(dim=-3)
    return torch.where(ancestor_mask(L.shape[-1], L.device), X, X.mT)


def foot_velocity_derivative(L, A, v):
    """
    d(L_c v) / dq in the tangent space, (..., 4, 3, nv), from the same
    kinematic Hessian: column j is a_j x (sum of v_i L_i over the joints i
    that j moves) + (sum of v_i a_i over the others) x L_j.
    """
    P = ancestor_mask(L.shape[-1], L.device).to(L.dtype)
    v = v[..., None, None, :]
    Lv = (L * v) @ P
    Av = (A * v) @ (1.0 - P)
    return torch.linalg.cross(A, Lv, dim=-2) + torch.linalg.cross(Av, L, dim=-2)


def split(x: torch.Tensor, u: torch.Tensor):
    """(normalized q, v, a, foot forces (..., 4, 3))"""
    return normalized(x[..., :19]), x[..., 19:], u[..., :18], u[..., 18:].unflatten(-1, (4, 3))


def inverse_dynamics(model, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Generalized forces tau = RNEA(q, v, a) - sum_c J_c^T f_c, (..., nv)."""
    q, v, a, f = split(x, u)
    _, L, _ = foot_kinematics(model, q)
    return gdyn.rnea(model, q, v, a) - torch.einsum("...cki,...ck->...i", L, f)


def inverse_dynamics_derivatives(model, x: torch.Tensor, u: torch.Tensor):
    """dtau/dx (..., nv, n_x) and dtau/du (..., nv, n_u)."""
    q, v, a, f = split(x, u)
    _, L, A = foot_kinematics(model, q)
    dtau_dq, dtau_dv, M = gdyn.rnea_derivatives(model, q, v, a)
    dtau_dq = (dtau_dq - jacobian_transpose_force_derivative(L, A, f)) @ tangent_map(
        x[..., :19]
    )
    dtau_dx = torch.cat([dtau_dq, dtau_dv], dim=-1)
    dtau_du = torch.cat([M, -L.flatten(-3, -2).mT], dim=-1)
    return dtau_dx, dtau_du


class Go2Dynamics(Dynamics):
    """
    Semi-implicit Euler step on the configuration manifold,
    v+ = v + dt a and q+ = integrate(q, dt v+). The foot forces in u only
    enter the inverse dynamics constraints.

    gdyn.d_integrate gives the Jacobians of integrate in the tangent spaces
    (Jq w.r.t. q, Jv w.r.t. the displacement); coordinate_map(q+) and
    tangent_map(q) turn them into Jacobians in the coordinates of x.
    """

    def __init__(self, model):
        super().__init__(nx=model.nq + model.nv, nu=model.nv + 12, nq=model.nq, nv=model.nv)
        self.model = model

    def f(self, x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
        v_next = x[..., self.nq :] + dt * u[..., : self.nv]
        q_next = gdyn.integrate(self.model, normalized(x[..., : self.nq]), dt * v_next)
        return torch.cat([q_next, v_next], dim=-1)

    def _integrate_jacobians(self, x: torch.Tensor, u: torch.Tensor, dt: float):
        """d q+ / d q (..., nq, nq) and d q+ / d (dt v+) (..., nq, nv), in coordinates."""
        q = normalized(x[..., : self.nq])
        dq = dt * (x[..., self.nq :] + dt * u[..., : self.nv])
        Jq, Jv = gdyn.d_integrate(self.model, q, dq)
        E = coordinate_map(gdyn.integrate(self.model, q, dq))
        return E @ Jq @ tangent_map(x[..., : self.nq]), E @ Jv

    def fx(self, x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
        dq_dq, dq_ddq = self._integrate_jacobians(x, u, dt)
        nq, nv = self.nq, self.nv
        jac = torch.zeros((*x.shape[:-1], self.nx, self.nx), dtype=x.dtype, device=x.device)
        jac[..., :nq, :nq] = dq_dq
        jac[..., :nq, nq:] = dt * dq_ddq
        jac[..., nq:, nq:] = torch.eye(nv, dtype=x.dtype, device=x.device)
        return jac

    def fu(self, x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
        _, dq_ddq = self._integrate_jacobians(x, u, dt)
        nq, nv = self.nq, self.nv
        jac = torch.zeros((*x.shape[:-1], self.nx, self.nu), dtype=x.dtype, device=x.device)
        jac[..., :nq, :nv] = dt * dt * dq_ddq
        jac[..., nq:, :nv] = dt * torch.eye(nv, dtype=x.dtype, device=x.device)
        return jac


class Go2BaseUnderactuation(UnderactuationConstraint):
    """The floating base is not actuated: tau[:6] = 0."""

    def __init__(self, model):
        super().__init__(n_h=6, n_x=model.nq + model.nv, n_u=model.nv + 12)
        self.model = model

    def h(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return inverse_dynamics(self.model, x, u)[..., :6]

    def hx(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return inverse_dynamics_derivatives(self.model, x, u)[0][..., :6, :]

    def hu(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return inverse_dynamics_derivatives(self.model, x, u)[1][..., :6, :]


class JointTorqueBounds(GenericConstraint):
    """Motor torque limits tau_lb <= tau[6:] <= tau_ub."""

    def __init__(self, model, lb: torch.Tensor, ub: torch.Tensor):
        super().__init__(
            n_g=model.nv - 6, n_x=model.nq + model.nv, n_u=model.nv + 12, lb=lb, ub=ub
        )
        self.model = model

    def g(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return inverse_dynamics(self.model, x, u)[..., 6:]

    def gx(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return inverse_dynamics_derivatives(self.model, x, u)[0][..., 6:, :]

    def gu(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return inverse_dynamics_derivatives(self.model, x, u)[1][..., 6:, :]


class FrictionCone(GenericConstraint):
    """
    Linearized friction cones of the foot forces (world frame, flat ground),
    5 rows per foot: f_z >= 0 and +-f_x, +-f_y + (mu / sqrt(2)) f_z >= 0
    (the pyramid inscribed in the cone). Linear in u.
    """

    def __init__(self, n_x: int, n_u: int, mu: float):
        n_g = 5 * len(FEET)
        super().__init__(
            n_g=n_g, n_x=n_x, n_u=n_u, lb=torch.zeros(n_g), ub=torch.full((n_g,), UNBOUNDED)
        )
        c = mu / math.sqrt(2.0)
        cone = torch.tensor(
            [[0.0, 0.0, 1.0], [1.0, 0.0, c], [-1.0, 0.0, c], [0.0, 1.0, c], [0.0, -1.0, c]]
        )
        self.G = torch.zeros((n_g, n_u))
        for k in range(len(FEET)):
            self.G[5 * k : 5 * k + 5, n_u - 12 + 3 * k : n_u - 9 + 3 * k] = cone

    def g(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return u @ self.G.T

    def gx(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return torch.zeros((*x.shape[:-1], self.n_g, self.n_x), dtype=x.dtype, device=x.device)

    def gu(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.G.expand(*u.shape[:-1], self.n_g, self.n_u)


class FootContact(GenericConstraint):
    """
    Sticking contacts: the feet stay at p0 (12 rows) with zero velocity
    J_c v (12 rows). Equalities (lb = ub = 0); p0 (B, 4, 3) per problem.
    """

    def __init__(self, model, p0: torch.Tensor):
        n_g = 6 * len(FEET)
        super().__init__(
            n_g=n_g,
            n_x=model.nq + model.nv,
            n_u=model.nv + 12,
            lb=torch.zeros(n_g),
            ub=torch.zeros(n_g),
        )
        self.model = model
        self.p0 = p0.flatten(-2)

    def g(self, x: torch.Tensor, u: torch.Tensor = None) -> torch.Tensor:
        q, v = normalized(x[..., :19]), x[..., 19:]
        p, L, _ = foot_kinematics(self.model, q)
        vel = torch.einsum("...cki,...i->...ck", L, v)
        return torch.cat([p.flatten(-2) - self.p0, vel.flatten(-2)], dim=-1)

    def gx(self, x: torch.Tensor, u: torch.Tensor = None) -> torch.Tensor:
        q, v = normalized(x[..., :19]), x[..., 19:]
        _, L, A = foot_kinematics(self.model, q)
        T = tangent_map(x[..., :19])
        L_flat = L.flatten(-3, -2)
        dp_dx = torch.cat([L_flat @ T, torch.zeros_like(L_flat)], dim=-1)
        dvel_dq = foot_velocity_derivative(L, A, v).flatten(-3, -2) @ T
        dvel_dx = torch.cat([dvel_dq, L_flat], dim=-1)
        return torch.cat([dp_dx, dvel_dx], dim=-2)

    def gu(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return torch.zeros((*x.shape[:-1], self.n_g, self.n_u), dtype=x.dtype, device=x.device)


def joint_limits(model):
    """Position, velocity and effort limits of the actuated joints from the
    URDF, in the model's joint order."""
    joints = {j.get("name"): j for j in ET.parse(URDF_PATH).getroot().iter("joint")}
    limits = [joints[name].find("limit") for name in model.names[2:]]
    return tuple(
        torch.tensor([float(lim.get(key)) for lim in limits])
        for key in ("lower", "upper", "velocity", "effort")
    )


def rpy_to_quat(rpy: torch.Tensor) -> torch.Tensor:
    """(x, y, z, w) quaternion of R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, cp, cy = torch.cos(0.5 * rpy).unbind(-1)
    sr, sp, sy = torch.sin(0.5 * rpy).unbind(-1)
    return torch.stack(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dim=-1,
    )


def pose_references(x_init: torch.Tensor, offset: torch.Tensor, horizon: int) -> torch.Tensor:
    """
    Reference states (B, H, n_x): the base moves from the stance by `offset`
    (B, 6: x, y, z, roll, pitch, yaw) along a minimum-jerk profile, so it
    starts and ends at rest. Joints at the stance, zero velocities.
    """
    s = torch.linspace(0.0, 1.0, horizon)
    phase = (10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5)[None, :, None]
    x_ref = x_init.repeat(offset.shape[0], horizon, 1)
    x_ref[..., :3] += phase * offset[:, None, :3]
    x_ref[..., 3:7] = rpy_to_quat(phase * offset[:, None, 3:])
    return x_ref


def main(args):
    # The default-device dispatch mode adds a large per-op overhead on CPU
    if args.device != "cpu":
        torch.set_default_device(args.device)
    batch_size = args.batch_size

    model = gdyn.load_urdf(URDF_PATH)
    assert model.nq == 19 and model.nv == 18, "expected the go2 free-flyer model"

    sqp_parameters = SqpParameters(
        **{
            ## ADMM ##
            "admm_backend": "bsqp",
            "admm_max_iter": 100,
            "admm_alpha": 1.6,
            "admm_sigma": 1e-6,
            # Rho related
            # The rho adapted to the previous QP slows the next one down
            "admm_reset_rho": True,
            "admm_update_rho": True,
            "admm_rho_init": 1e-2,
            "admm_rho_min": 1e-6,
            "admm_rho_max": 1e8,
            "admm_adaptive_rho_tolerance": 10.0,
            "admm_rho_update_iter_freq": 5,
            # Warm starting
            "admm_warm_start_unconstrained": True,
            "admm_reset_ksi": False,
            # Tolerances
            "admm_abs_tolerance": 0.001,
            "admm_abs_tolerance_final": -1.0,
            "admm_rel_tolerance": 0.0001,
            "admm_rel_tolerance_final": -1.0,
            "admm_tolerance_update_steps": 0,
            ## SQP ##
            "sqp_max_iter": 30,
            "lqr_reg_init": 1e-5,
            "merit_mu": 1e4,
            "armijo_beta": 1e-4,
            "ls_max_iter": 10,
            # Steps of the inexact (ADMM) QPs stay around 0.1 in the accelerations
            "sqp_cost_eps": 1e-1,
            "sqp_viol_eps": 1e-3,
            "check_complementarity": False,
            "qp_solver": "lqr",
            "ls_function": "merit_adaptive",
            "second_order_correction": True,
        }
    )

    # Nominal stance (hip, thigh, calf per leg), base height with the feet at z = 0
    q_joints = [0.0, 0.8, -1.6] * 4
    q_stance = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] + q_joints)
    q_stance[2] = -gdyn.forward_kinematics(model, q_stance, FEET)[:, 2, 3].mean()

    q_lb, q_ub, v_max, tau_max = joint_limits(model)
    free = [UNBOUNDED] * 7
    free_v = [UNBOUNDED] * 6
    a_max = [50.0] * 6 + [100.0] * 12

    x_init = q_stance.tolist() + [0.0] * 18
    problem_parameters = ProblemParameters(
        **{
            "inverse_dynamics": True,
            "n_h": 6,  # Unactuated floating base
            "batch_size": batch_size,
            "dt": 0.02,
            "tf": 1.0,
            "x_init": x_init,
            "x_des": x_init,
            # The contacts fix the initial state: no perturbation
            "noise_std": [0.0] * 37,
            # Joint position (rad) and velocity (rad/s) limits, free base
            "x_lb": [-b for b in free] + q_lb.tolist() + [-b for b in free_v] + (-v_max).tolist(),
            "x_ub": free + q_ub.tolist() + free_v + v_max.tolist(),
            # Accelerations; the friction cones bound the forces
            "u_lb": [-a for a in a_max] + [-UNBOUNDED] * 12,
            "u_ub": a_max + [UNBOUNDED] * 12,
            # Base pose tracking, posture and velocity regularization
            "q_w": [1e3] * 7 + [1e-1] * 12 + [1e-1] * 6 + [1e-2] * 12,
            # Accelerations, forces around the static ones. Weights well below
            # rho (forces 1e-5) make ADMM converge in hundreds of iterations
            "r_w": [1e-3] * 18 + [1e-3] * 12,
            # Terminal: reach the pose at rest
            "qf_w": [1e3] * 7 + [1e-1] * 12 + [1e1] * 18,
        }
    )

    system_parameters = Go2Parameters()
    problem = Problem(problem_parameters, system_parameters)
    problem.dynamics = Go2Dynamics(model)
    problem.underactuation = Go2BaseUnderactuation(model)

    # Reference: move the base by a random pose offset (x, y, z, roll, pitch, yaw)
    offset_max = torch.tensor([0.03, 0.03, 0.04, 0.15, 0.15, 0.15])
    offset = (2.0 * torch.rand(batch_size, 6) - 1.0) * offset_max
    x_ref = pose_references(problem_parameters.x_init, offset, problem.horizon)
    x_des = x_ref[:, -1]

    # Static foot forces at the stance (minimum norm, base rows of the inverse dynamics)
    q0 = q_stance.unsqueeze(0)
    p0, L0, _ = foot_kinematics(model, q0)
    gravity = gdyn.rnea(model, q0, torch.zeros(1, 18), torch.zeros(1, 18))
    f_static = (torch.linalg.pinv(L0.flatten(-3, -2).mT[:, :6]) @ gravity[:, :6, None]).squeeze(-1)
    u_static = torch.cat([torch.zeros(1, 18), f_static], dim=-1).repeat(batch_size, 1)

    Q = problem_parameters.q_w * torch.eye(problem.n_x).repeat(batch_size, 1, 1)
    R = problem_parameters.r_w * torch.eye(problem.n_u).repeat(batch_size, 1, 1)
    Qf = problem_parameters.qf_w * torch.eye(problem.n_x).repeat(batch_size, 1, 1)
    stage_costs = [
        LqrCost(Q=Q, R=R, x_des=x_ref[:, k], u_des=u_static)
        for k in range(problem.horizon - 1)
    ]
    terminal_cost = LqrCost(Q=Qf, x_des=x_des)

    state_bounds = StateBounds(
        problem.n_x, problem.n_u, problem_parameters.x_lb, problem_parameters.x_ub
    )
    control_bounds = ControlBounds(
        problem.n_x, problem.n_u, problem_parameters.u_lb, problem_parameters.u_ub
    )
    torque_bounds = JointTorqueBounds(model, -tau_max, tau_max)
    friction_cone = FrictionCone(problem.n_x, problem.n_u, mu=0.8)
    contact = FootContact(model, p0.repeat(batch_size, 1, 1))

    for k in range(problem.horizon - 1):
        problem.costs.append([stage_costs[k]])
        problem.constraints[k] = [
            state_bounds,
            control_bounds,
            torque_bounds,
            friction_cone,
            contact,
        ]
    problem.costs.append([terminal_cost])
    problem.constraints[-1] = [state_bounds, contact]

    # Initial guess: stand still on the static forces (dynamically consistent)
    initial_guess = SqpSolution(
        x=problem_parameters.x_init.repeat(batch_size, problem.horizon, 1),
        u=u_static.unsqueeze(1).repeat(1, problem.horizon - 1, 1),
        mu=torch.zeros((batch_size, problem.horizon, problem.n_x)),
        nu=torch.zeros((batch_size, problem.horizon - 1, problem.n_h)),
        ksi=[None] * problem.horizon,
    )

    print(f"Solving {batch_size} Go2 standing problems (base rows in the Riccati)...")
    solution, log = sqp_solve(problem, sqp_parameters, initial_guess)
    print(log)
    report(model, solution, log, x_des, p0, tau_max, friction_cone)

    if args.save:
        save_solution(solution, args.save, x_des=x_des)


def report(model, solution, log, x_des, p0, tau_max, friction_cone):
    x, u = solution.x, solution.u
    q = normalized(x[..., :19])

    pos_error = torch.norm(q[:, -1, :3] - x_des[:, :3], dim=-1)
    quat_dot = torch.sum(q[:, -1, 3:7] * x_des[:, 3:7], dim=-1).abs().clamp(max=1.0)
    ori_error = 2.0 * torch.acos(quat_dot)
    print(f"Final base position error (max over batch): {pos_error.max().item():.2e} m")
    print(f"Final base rotation error (max over batch): {ori_error.max().item():.2e} rad")

    p, L, _ = foot_kinematics(model, q)
    drift = torch.norm(p - p0, dim=-1).max()
    speed = torch.norm(torch.einsum("...cki,...i->...ck", L, x[..., 19:]), dim=-1).max()
    print(f"Foot drift / speed      (max over batch): {drift.item():.2e} m / {speed.item():.2e} m/s")

    tau = inverse_dynamics(model, x[:, :-1], u)
    print(f"Base wrench |tau[:6]|   (max over batch): {tau[..., :6].abs().max().item():.2e}")
    print(f"Peak |tau| / tau_max    (max over batch): {(tau[..., 6:].abs() / tau_max).max().item():.2f}")
    margin = friction_cone.g(x[:, :-1], u).min()
    print(f"Friction cone margin    (min over batch): {margin.item():.2e} N")
    print(f"Constraint violation    (max over batch): {max(log.constraint_violation):.2e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="Number of problems")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--save", type=str, help="Save path (without .pt)")
    main(parser.parse_args())
