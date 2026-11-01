"""Batched Franka (FP3) end-effector reaching with joint and velocity limits.

Shows how to plug user-defined models into diffsqp:
  * `KinematicDynamics` subclasses `Dynamics` with x_dot = u (joint velocities).
  * `EndEffectorTrackingCost` subclasses `Cost` with a 6D pose error computed
    with bard forward kinematics, using a Gauss-Newton Hessian J^T Q J.
Joint position limits (`StateBounds`) and joint velocity limits
(`ControlBounds`) are taken from the FP3 datasheet.

    uv run examples/franka_constrained.py --batch_size 16 --save out/franka
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import bard

from diffsqp.problems import Problem, ProblemParameters
from diffsqp.costs import LqrCost, Cost
from diffsqp.solvers import sqp_solve, SqpParameters
from diffsqp.dynamics import Dynamics
from diffsqp.constraints import StateBounds, ControlBounds
from diffsqp.types import SqpSolution
from diffsqp.utils.load_save import save_solution

URDF_PATH = Path(__file__).resolve().parent.parent / "resources" / "robots" / "fp3.urdf"


@dataclass
class KinematicDynamicsParameters:
    name: str = "bard_kinematics"
    n_x: int = 7
    n_u: int = 7
    n_q: int = 7
    n_v: int = 0


class KinematicDynamics(Dynamics):
    """Continuous time kinematic dynamics: x_dot = u"""

    def __init__(self, parameters: KinematicDynamicsParameters):
        super().__init__(
            nx=parameters.n_x,
            nu=parameters.n_u,
            nq=parameters.n_q,
            nv=parameters.n_v,
            use_semi_implicit=False,
        )

    def fc(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Continuous time dynamics: x_dot = u"""
        return u

    def fcx(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """dfc/dx matrix: Zeros"""
        batch_size = x.shape[0]
        return torch.zeros(
            (batch_size, self.nx, self.nx), device=x.device, dtype=x.dtype
        )

    def fcu(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """dfc/du matrix: Identity"""
        batch_size = x.shape[0]
        I = torch.eye(self.nu, device=x.device, dtype=x.dtype)
        return I.expand(batch_size, self.nx, self.nu)


class EndEffectorTrackingCost(Cost):
    """
    Weighted squared pose error cost utilizing bard forward kinematics.
    Tracks both 3D position and 3D orientation.
    """

    def __init__(
        self, model, data, eef_id: int, T_ref: torch.Tensor, Q_diag: torch.Tensor
    ):
        super().__init__()
        self.model = model
        self.data = data
        self.eef_id = eef_id
        self.T_ref = T_ref
        self.Q_diag = Q_diag

    def compute_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the 6D error vector [pos_err(3), ori_err(3)] between the current transform and reference.
        """
        transforms = bard.forward_kinematics(self.model, self.data, self.eef_id, q=x)
        T_ref = self.T_ref.to(device=x.device, dtype=x.dtype)

        # Position error (3D)
        p_ee = transforms[..., :3, 3]
        p_ref = T_ref[..., :3, 3]
        pos_err = p_ee - p_ref

        # Orientation error using SO(3) skew-symmetric mapping
        R_ee = transforms[..., :3, :3]
        R_ref = T_ref[..., :3, :3]

        # e_R = 0.5 * vee(R_ee * R_ref^T - R_ref * R_ee^T)
        R_ref_T = R_ref.transpose(-1, -2)
        R_ee_T = R_ee.transpose(-1, -2)
        S = torch.matmul(R_ee, R_ref_T) - torch.matmul(R_ref, R_ee_T)

        # Extract the vector part (vee operator) from the skew-symmetric matrix S
        ori_err = 0.5 * torch.stack([S[..., 2, 1], S[..., 0, 2], S[..., 1, 0]], dim=-1)

        return torch.cat([pos_err, ori_err], dim=-1)

    def l(self, x: torch.Tensor, u: torch.Tensor = None) -> torch.Tensor:
        err = self.compute_error(x)
        Q = self.Q_diag.to(device=x.device, dtype=x.dtype)

        cost = 0.5 * torch.sum(Q * (err**2), dim=-1)
        return cost

    def lx(self, x: torch.Tensor, u: torch.Tensor = None) -> torch.Tensor:
        err = self.compute_error(x)
        Q = self.Q_diag.to(device=x.device, dtype=x.dtype)

        qd_dummy = torch.zeros_like(x)
        bard.update_kinematics(self.model, self.data, x, qd_dummy)

        # Full 6D spatial Jacobian
        J = bard.jacobian(self.model, self.data, self.eef_id, reference_frame="world")

        # Assuming bard Jacobian returns [v; w] in the first 6 rows
        J_full = J[..., :6, :]
        return torch.einsum("...ij,...i,...i->...j", J_full, Q, err)

    def lxx(self, x: torch.Tensor, u: torch.Tensor = None) -> torch.Tensor:
        qd_dummy = torch.zeros_like(x)
        bard.update_kinematics(self.model, self.data, x, qd_dummy)

        J = bard.jacobian(self.model, self.data, self.eef_id, reference_frame="world")
        J_full = J[..., :6, :]

        Q = self.Q_diag.to(device=x.device, dtype=x.dtype)
        Q_J = Q.unsqueeze(-1) * J_full

        # Gauss-Newton Hessian Approximation: J^T * Q * J
        return torch.einsum("...mi,...mk->...ik", J_full, Q_J)

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


def main(args):
    torch.set_default_device(args.device)
    batch_size = args.batch_size

    model = bard.build_model_from_urdf(URDF_PATH, floating_base=False)
    model.to(dtype=torch.float32, device=args.device)
    data = bard.create_data(model, max_batch_size=batch_size)

    sqp_parameters = SqpParameters(
        **{
            ## ADMM ##
            "admm_max_iter": 10,
            "admm_alpha": 1.6,
            "admm_sigma": 1e-6,
            # Rho related
            "admm_reset_rho": False,
            "admm_update_rho": False,
            "admm_rho_init": 0.1,
            "admm_rho_min": 1e-6,
            "admm_rho_max": 10.0,
            "admm_adaptive_rho_tolerance": 2.0,
            "admm_rho_update_iter_freq": 25,
            # Warm starting
            "admm_warm_start_unconstrained": False,
            "admm_reset_ksi": False,
            # Tolerances
            "admm_abs_tolerance": 0.1,
            "admm_abs_tolerance_final": -1.0,
            "admm_rel_tolerance": 0.01,
            "admm_rel_tolerance_final": -1.0,
            "admm_tolerance_update_steps": 0,
            ## SQP ##
            "sqp_max_iter": 5,
            "lqr_reg_init": 1e0,
            "merit_mu": 1e4,
            "armijo_beta": 1e0,
            "ls_max_iter": 10,
            "sqp_cost_eps": 1e-1,
            "sqp_viol_eps": 1e-2,
            "check_complementarity": False,
            "qp_solver": "lqr",
            "ls_function": "filter",
        }
    )

    problem_parameters = ProblemParameters(
        **{
            "inverse_dynamics": False,
            "n_h": 0,
            "batch_size": batch_size,
            "dt": 0.01,
            "tf": 1.0,
            "x_init": [0.0, 0.0, 0.0, -torch.pi / 2.0, 0.0, torch.pi / 2.0, 0.0],
            "x_des": [0.0, 0.0, 0.2, -torch.pi / 2.0, 0.0, torch.pi / 2.0, 0.0],
            "noise_std": [0.05] * 7,
            # Joint position (rad) and velocity (rad/s) limits
            "x_lb": [-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508],
            "x_ub": [2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508],
            "u_lb": [-2.62, -2.62, -2.62, -2.62, -5.26, -4.18, -5.26],
            "u_ub": [2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26],
            "q_w": [1e-8] * 7,
            "r_w": [1e-6] * 7,
            "qf_w": [1e1] * 7,  # Unused: the terminal cost is the EE tracking cost
        }
    )

    system_parameters = KinematicDynamicsParameters()
    dynamics = KinematicDynamics(system_parameters)
    problem = Problem(problem_parameters, system_parameters)
    problem.dynamics = dynamics

    x_refs = torch.tensor(
        [
            [0.2800, 0.3000, 0.0000, -1.7900, 0.0000, 2.1200, 0.0000],
            [-0.0400, -0.1700, -0.2600, -1.5708, -0.3600, 1.2700, 0.0000],
            [0.0000, 0.0000, -0.3100, -1.8400, -0.5700, 1.5708, 0.0000],
            [-0.2600, 0.3000, 0.0000, -1.8100, 0.0000, 2.1500, 0.0000],
        ]
    )
    T_refs = torch.tensor(
        [
            [
                [9.6062e-01, 2.7636e-01, 2.8827e-02, 6.0978e-01],
                [2.7623e-01, -9.6106e-01, 8.2894e-03, 1.7534e-01],
                [2.9995e-02, -8.4831e-16, -9.9955e-01, 4.9424e-01],
                [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
            ],
            [
                [0.9156, -0.3327, -0.2260, 0.4524],
                [-0.3833, -0.8919, -0.2401, -0.1686],
                [-0.1217, 0.3065, -0.9441, 0.7913],
                [0.0000, 0.0000, 0.0000, 1.0000],
            ],
            [
                [0.9180, -0.1201, -0.3779, 0.5328],
                [-0.2941, -0.8456, -0.4456, -0.1707],
                [-0.2660, 0.5202, -0.8116, 0.6030],
                [0.0000, 0.0000, 0.0000, 1.0000],
            ],
            [
                [9.6562e-01, -2.5708e-01, 3.8646e-02, 6.1076e-01],
                [-2.5687e-01, -9.6639e-01, -1.0281e-02, -1.6248e-01],
                [3.9990e-02, -8.4815e-16, -9.9920e-01, 4.8765e-01],
                [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
            ],
        ],
    )

    target_choices = torch.randint(0, 4, (batch_size,))

    T_ref_batch = T_refs[target_choices]
    x_ref_batch = x_refs[target_choices]

    Q_ee_diag = torch.tensor([1e5, 1e5, 1e5, 1e5, 1e5, 1e5])
    eef_id = model.get_frame_id("fp3_link7")
    ee_cost = EndEffectorTrackingCost(model, data, eef_id, T_ref_batch, Q_ee_diag)

    # Regularize towards the start configuration and penalize joint velocities
    Q = problem_parameters.q_w * torch.eye(dynamics.nx).repeat(problem.batch_size, 1, 1)
    R = problem_parameters.r_w * torch.eye(dynamics.nu).repeat(problem.batch_size, 1, 1)
    Qf = problem_parameters.qf_w * torch.eye(dynamics.nx).repeat(
        problem.batch_size, 1, 1
    )
    reg_cost = LqrCost(Q=Q, R=R, x_des=problem_parameters.x_init.detach().clone())

    state_bounds = StateBounds(
        problem.n_x, problem.n_u, problem_parameters.x_lb, problem_parameters.x_ub
    )
    control_bounds = ControlBounds(
        problem.n_x, problem.n_u, problem_parameters.u_lb, problem_parameters.u_ub
    )

    for k in range(problem.horizon - 1):
        problem.costs.append([reg_cost])
        problem.constraints[k] = [state_bounds, control_bounds]
    problem.costs.append([ee_cost])
    problem.constraints[-1] = [state_bounds]

    # Initial guess: stay at the (perturbed) start configuration
    initial_guess = SqpSolution(
        x=problem_parameters.x_init.repeat(batch_size, problem.horizon, 1),
        u=torch.zeros((batch_size, problem.horizon - 1, problem.n_u)),
        mu=torch.zeros((batch_size, problem.horizon, problem.n_x)),
        nu=torch.zeros((batch_size, problem.horizon - 1, problem.n_h)),
        ksi=[None] * problem.horizon,
    )
    noise_std = torch.tensor(problem_parameters.noise_std)
    initial_guess.x[:, 0] += noise_std * torch.randn((batch_size, len(noise_std)))

    print(f"Solving {batch_size} Franka end-effector reaching problems...")
    solution, log = sqp_solve(problem, sqp_parameters, initial_guess)
    print(log)

    T_final = bard.forward_kinematics(model, data, eef_id, q=solution.x[:, -1])
    pos_error = torch.norm(T_final[:, :3, 3] - T_ref_batch[:, :3, 3], dim=-1)
    print(f"Final EE position error (max over batch): {pos_error.max().item():.2e} m")
    print(f"Constraint violation (max over batch): {max(log.constraint_violation):.2e}")

    if args.save:
        save_solution(solution, args.save, x_des=x_ref_batch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="Number of problems")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--save", type=str, help="Save path (without .pt)")
    main(parser.parse_args())
