"""Batched Franka (FP3) SE(3) end-effector tracking with inverse dynamics.

The state is x = [q, v] (joint positions and velocities) and the controls are
the joint accelerations u = a, integrated by the generic `Dynamics` (a
semi-implicit double integrator). The joint torques follow from inverse
dynamics, tau = RNEA(q, v, a), computed with gdyn. `TorqueBounds` bounds them,
linearized with the analytical RNEA derivatives (dtau/dq, dtau/dv and
dtau/da = M(q)). The arm is fully actuated, so there is no underactuation
constraint (n_h = 0).

`SE3TrackingCost` tracks a time-varying flange pose at every stage: the
position error and the rotation vector log(R R_ref^T), both in world axes,
with a Gauss-Newton Hessian from the gdyn frame Jacobian. The reference moves
the flange around a horizontal circle while rotating it about a random axis,
and starts and ends at rest at the initial pose.

    uv run examples/franka_se3_tracking.py --batch_size 16 --save out/franka_tracking
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import gdyn

from diffsqp.problems import Problem, ProblemParameters
from diffsqp.costs import LqrCost, Cost
from diffsqp.solvers import sqp_solve, SqpParameters
from diffsqp.dynamics import Dynamics
from diffsqp.constraints import GenericConstraint, StateBounds, ControlBounds
from diffsqp.types import SqpSolution
from diffsqp.utils.load_save import save_solution

URDF_PATH = Path(__file__).resolve().parent.parent / "resources" / "robots" / "fp3.urdf"


@dataclass
class FrankaParameters:
    name: str = "fp3"
    n_x: int = 14  # [q, v]
    n_u: int = 7  # Joint accelerations
    n_q: int = 7
    n_v: int = 7


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


def so3_log(R: torch.Tensor) -> torch.Tensor:
    """Rotation vector w with exp(skew(w)) = R, for rotation angles below pi."""
    vee = torch.stack(
        [R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]],
        dim=-1,
    )
    sin = 0.5 * torch.norm(vee, dim=-1)
    cos = 0.5 * (R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0)
    theta = torch.atan2(sin, cos)

    # theta / (2 sin(theta)), with its Taylor expansion near 0
    small = theta < 1e-4
    scale = torch.where(
        small, 0.5 + theta**2 / 12.0, theta / (2.0 * torch.where(small, 1.0, sin))
    )
    return scale.unsqueeze(-1) * vee


def so3_left_jacobian_inv(w: torch.Tensor) -> torch.Tensor:
    """Inverse left Jacobian of SO(3): log(exp(d) exp(w)) = w + Jl^-1(w) d + O(|d|^2)."""
    theta2 = torch.sum(w * w, dim=-1)
    theta = torch.sqrt(theta2)

    # 1 / theta^2 - (1 + cos) / (2 theta sin). The difference cancels badly in
    # float32 for small angles, so use the Taylor expansion below 0.1 rad
    small = theta < 0.1
    t = torch.where(small, 1.0, theta)
    c = torch.where(
        small,
        1.0 / 12.0 + theta2 / 720.0,
        1.0 / t**2 - (1.0 + torch.cos(t)) / (2.0 * t * torch.sin(t)),
    )

    W = skew(w)
    I = torch.eye(3, device=w.device, dtype=w.dtype)
    return I - 0.5 * W + c[..., None, None] * (W @ W)


class TorqueBounds(GenericConstraint):
    """Joint torque limits tau_lb <= RNEA(q, v, a) <= tau_ub, with x = [q, v], u = a."""

    def __init__(self, model, lb: torch.Tensor, ub: torch.Tensor):
        super().__init__(
            n_g=model.nv, n_x=model.nq + model.nv, n_u=model.nv, lb=lb, ub=ub
        )
        self.model = model

    def g(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        nq = self.model.nq
        return gdyn.rnea(self.model, x[..., :nq], x[..., nq:], u)

    def gx(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        nq = self.model.nq
        dtau_dq, dtau_dv, _ = gdyn.rnea_derivatives(
            self.model, x[..., :nq], x[..., nq:], u
        )
        return torch.cat([dtau_dq, dtau_dv], dim=-1)

    def gu(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """dtau/da: the joint-space mass matrix M(q)"""
        nq = self.model.nq
        _, _, M = gdyn.rnea_derivatives(self.model, x[..., :nq], x[..., nq:], u)
        return M


class SE3TrackingCost(Cost):
    """
    Weighted squared pose error of a frame, 0.5 * sum(W * e^2), with
    e = [p - p_ref, log(R R_ref^T)] (position and rotation vector, in world axes).

    The Hessian is the Gauss-Newton approximation J_e^T W J_e, with
    J_e = [J_lin; Jl^-1(e_R) J_ang] built from the LOCAL_WORLD_ALIGNED frame
    Jacobian, whose rows are the velocity of the frame origin and the angular
    velocity of the frame, both in world axes. Only depends on q = x[:nq].
    """

    def __init__(self, model, frame_id: int, T_ref: torch.Tensor, W_diag: torch.Tensor):
        super().__init__()
        self.model = model
        self.frame_id = frame_id
        self.T_ref = T_ref
        self.W_diag = W_diag

    def residual(self, oMf: torch.Tensor) -> torch.Tensor:
        pos_err = oMf[..., :3, 3] - self.T_ref[..., :3, 3]
        ori_err = so3_log(oMf[..., :3, :3] @ self.T_ref[..., :3, :3].mT)
        return torch.cat([pos_err, ori_err], dim=-1)

    def residual_jacobian(self, x: torch.Tensor):
        """Residual e (B, 6) and its Jacobian de/dx (B, 6, n_x)."""
        nq, nv = self.model.nq, self.model.nv
        J, oMf = gdyn.frame_jacobians(
            self.model,
            x[..., :nq],
            self.frame_id,
            gdyn.LOCAL_WORLD_ALIGNED,
            return_placements=True,
        )
        e = self.residual(oMf)

        J_e = torch.zeros((*x.shape[:-1], 6, x.shape[-1]), device=x.device, dtype=x.dtype)
        J_e[..., :3, :nv] = J[..., :3, :]
        J_e[..., 3:, :nv] = so3_left_jacobian_inv(e[..., 3:]) @ J[..., 3:, :]
        return e, J_e

    def l(self, x: torch.Tensor, u: torch.Tensor = None) -> torch.Tensor:
        oMf = gdyn.forward_kinematics(self.model, x[..., : self.model.nq], self.frame_id)
        e = self.residual(oMf)
        return 0.5 * torch.sum(self.W_diag * e**2, dim=-1)

    def lx(self, x: torch.Tensor, u: torch.Tensor = None) -> torch.Tensor:
        e, J_e = self.residual_jacobian(x)
        return torch.einsum("...ij,...i->...j", J_e, self.W_diag * e)

    def lxx(self, x: torch.Tensor, u: torch.Tensor = None) -> torch.Tensor:
        # Gauss-Newton Hessian Approximation: J^T * W * J
        _, J_e = self.residual_jacobian(x)
        return J_e.mT @ (self.W_diag.unsqueeze(-1) * J_e)

    def lu(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(u)

    def luu(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (*u.shape[:-1], u.shape[-1], u.shape[-1]), device=u.device, dtype=u.dtype
        )

    def lux(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (*u.shape[:-1], u.shape[-1], x.shape[-1]), device=u.device, dtype=u.dtype
        )

    def lxu(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (*u.shape[:-1], x.shape[-1], u.shape[-1]), device=u.device, dtype=u.dtype
        )


def circle_references(
    T0: torch.Tensor, radius: torch.Tensor, axis: torch.Tensor, angle: float, horizon: int
) -> torch.Tensor:
    """
    Reference poses (B, H, 4, 4): a horizontal circle of `radius` through the
    start position, while rotating by angle * sin(phase) about the unit `axis`
    (world axes). The phase follows a minimum-jerk profile from 0 to 2 pi, so
    the reference starts and ends at rest at T0.
    """
    s = torch.linspace(0.0, 1.0, horizon)
    phase = 2.0 * math.pi * (10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5)

    circle = torch.stack(
        [torch.cos(phase) - 1.0, torch.sin(phase), torch.zeros_like(phase)], dim=-1
    )
    rotvec = angle * torch.sin(phase)[:, None] * axis[:, None, :]

    T = T0.repeat(radius.shape[0], horizon, 1, 1)
    T[..., :3, 3] += radius[:, None, None] * circle
    T[..., :3, :3] = torch.linalg.matrix_exp(skew(rotvec)) @ T0[:3, :3]
    return T


def main(args):
    # The default-device dispatch mode adds a large per-op overhead on CPU
    if args.device != "cpu":
        torch.set_default_device(args.device)
    batch_size = args.batch_size

    model = gdyn.load_urdf(URDF_PATH)
    flange_id = model.get_frame_id("fp3_link8")

    sqp_parameters = SqpParameters(
        **{
            ## ADMM ##
            "admm_max_iter": 30,
            "admm_alpha": 1.6,
            "admm_sigma": 1e-6,
            # Rho related
            "admm_reset_rho": False,
            "admm_update_rho": True,
            "admm_rho_init": 0.1,
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
            "sqp_cost_eps": 1e-2,
            "sqp_viol_eps": 1e-3,
            "check_complementarity": False,
            "qp_solver": "lqr",
            "ls_function": "merit_adaptive",
            "second_order_correction": True,
        }
    )

    v_max = [2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26]
    a_max = [10.0] * 7
    tau_max = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]

    # State: [joint positions (7), joint velocities (7)]
    problem_parameters = ProblemParameters(
        **{
            "inverse_dynamics": True,
            "n_h": 0,  # Fully actuated
            "batch_size": batch_size,
            "dt": 0.02,
            "tf": 2.0,
            "x_init": [0.0, 0.0, 0.0, -torch.pi / 2.0, 0.0, torch.pi / 2.0, 0.0]
            + [0.0] * 7,
            # The reference returns to the start pose
            "x_des": [0.0, 0.0, 0.0, -torch.pi / 2.0, 0.0, torch.pi / 2.0, 0.0]
            + [0.0] * 7,
            "noise_std": [0.02] * 7 + [0.0] * 7,
            # Joint position (rad) and velocity (rad/s) limits
            "x_lb": [-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508]
            + [-v for v in v_max],
            "x_ub": [2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508] + v_max,
            # Joint acceleration limits (rad/s^2)
            "u_lb": [-a for a in a_max],
            "u_ub": a_max,
            # Posture and velocity regularization, acceleration effort
            "q_w": [1e-2] * 7 + [1e-3] * 7,
            "r_w": [1e-4] * 7,
            # Terminal: come to rest
            "qf_w": [1e-2] * 7 + [1e1] * 7,
        }
    )

    system_parameters = FrankaParameters()
    # Integrator only: the physics lives in the torque constraint
    dynamics = Dynamics(
        nx=system_parameters.n_x,
        nu=system_parameters.n_u,
        nq=system_parameters.n_q,
        nv=system_parameters.n_v,
    )
    problem = Problem(problem_parameters, system_parameters)
    problem.dynamics = dynamics

    # Flange reference: circles of random radius, rotating about random axes
    x_init = problem_parameters.x_init
    T0 = gdyn.forward_kinematics(model, x_init[: model.nq], flange_id)
    radius = 0.05 + 0.07 * torch.rand(batch_size)
    axis = torch.nn.functional.normalize(torch.randn(batch_size, 3), dim=-1)
    T_ref = circle_references(T0, radius, axis, angle=0.4, horizon=problem.horizon)

    W_diag = torch.tensor([1e3, 1e3, 1e3, 1e2, 1e2, 1e2])
    ee_costs = [
        SE3TrackingCost(model, flange_id, T_ref[:, k], W_diag)
        for k in range(problem.horizon)
    ]

    Q = problem_parameters.q_w * torch.eye(dynamics.nx).repeat(batch_size, 1, 1)
    R = problem_parameters.r_w * torch.eye(dynamics.nu).repeat(batch_size, 1, 1)
    Qf = problem_parameters.qf_w * torch.eye(dynamics.nx).repeat(batch_size, 1, 1)
    reg_cost = LqrCost(Q=Q, R=R, x_des=x_init.clone())
    terminal_cost = LqrCost(Q=Qf, x_des=problem_parameters.x_des.clone())

    state_bounds = StateBounds(
        problem.n_x, problem.n_u, problem_parameters.x_lb, problem_parameters.x_ub
    )
    control_bounds = ControlBounds(
        problem.n_x, problem.n_u, problem_parameters.u_lb, problem_parameters.u_ub
    )
    tau_ub = torch.tensor(tau_max)
    torque_bounds = TorqueBounds(model, -tau_ub, tau_ub)

    for k in range(problem.horizon - 1):
        problem.costs.append([ee_costs[k], reg_cost])
        problem.constraints[k] = [state_bounds, control_bounds, torque_bounds]
    problem.costs.append([ee_costs[-1], terminal_cost])
    problem.constraints[-1] = [state_bounds]

    # Initial guess: stay at rest at the (perturbed) start configuration
    initial_guess = SqpSolution(
        x=x_init.repeat(batch_size, problem.horizon, 1),
        u=torch.zeros((batch_size, problem.horizon - 1, problem.n_u)),
        mu=torch.zeros((batch_size, problem.horizon, problem.n_x)),
        nu=torch.zeros((batch_size, problem.horizon - 1, problem.n_h)),
        ksi=[None] * problem.horizon,
    )
    noise_std = torch.tensor(problem_parameters.noise_std)
    initial_guess.x[:, 0] += noise_std * torch.randn((batch_size, len(noise_std)))

    print(f"Solving {batch_size} Franka SE(3) tracking problems (inverse dynamics)...")
    solution, log = sqp_solve(problem, sqp_parameters, initial_guess)
    print(log)

    # Tracking error along the trajectory. The first stages inherit the offset
    # of the perturbed start, so report the mean and the final error
    nq = model.nq
    oMf = gdyn.forward_kinematics(model, solution.x[..., :nq], flange_id)
    pos_error = torch.norm(oMf[..., :3, 3] - T_ref[..., :3, 3], dim=-1)
    ori_error = torch.norm(so3_log(oMf[..., :3, :3] @ T_ref[..., :3, :3].mT), dim=-1)
    print(
        f"Flange position error  (mean / final, max over batch): "
        f"{pos_error.mean().item():.2e} / {pos_error[:, -1].max().item():.2e} m"
    )
    print(
        f"Flange rotation error  (mean / final, max over batch): "
        f"{ori_error.mean().item():.2e} / {ori_error[:, -1].max().item():.2e} rad"
    )

    tau = gdyn.rnea(
        model, solution.x[:, :-1, :nq], solution.x[:, :-1, nq:], solution.u
    )
    print(f"Peak |tau| / tau_max   (max over batch): {(tau.abs() / tau_ub).max().item():.2f}")
    print(f"Constraint violation   (max over batch): {max(log.constraint_violation):.2e}")

    if args.save:
        save_solution(solution, args.save, x_des=problem_parameters.x_des)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="Number of problems")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--save", type=str, help="Save path (without .pt)")
    main(parser.parse_args())
