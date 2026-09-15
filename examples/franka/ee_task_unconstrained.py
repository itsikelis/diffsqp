import argparse
import torch
import bard

from diffsqp.problems import Problem, ProblemParameters
from diffsqp.costs import LqrCost, Cost
from diffsqp.solvers import sqp_solve, SqpParameters
from diffsqp.dynamics.base_dynamics import Dynamics
from diffsqp.constraints import StateBounds, ControlBounds
from diffsqp.types import SqpSolution

from dataclasses import dataclass
from pathlib import Path

from diffsqp.utils.load_save import *

##########################
# Custom Kinematics Task #
##########################


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
    device = args.device
    torch.set_default_device(device)
    batch_size = args.batch_size

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    urdf_path = BASE_DIR / "resources" / "robots" / "fp3.urdf"
    model = bard.build_model_from_urdf(urdf_path, floating_base=False)
    model.to(dtype=torch.float32, device=device)
    data = bard.create_data(model, max_batch_size=batch_size)

    sqp_parameters = SqpParameters(
        **{
            ## ADMM ##
            "admm_max_iter": 50,
            "admm_alpha": 1.6,
            "admm_sigma": 1e-6,
            # Rho related
            "admm_reset_rho": False,
            "admm_update_rho": True,
            "admm_rho_init": 0.8,
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
            "ls_function": "merit",
        }
    )

    problem_parameters = ProblemParameters(
        **{
            "inverse_dynamics": False,
            "n_h": 0,
            "batch_size": batch_size,
            "dt": 0.005,
            "tf": 0.3,
            "x_init": [0.0, 0.0, 0.0, -torch.pi / 2.0, 0.0, torch.pi / 2.0, 0.0],
            "x_des": [0.0, 0.0, 0.2, -torch.pi / 2.0, 0.0, torch.pi / 2.0, 0.0],
            "noise_std": [0.05] * 7,
            "x_lb": [-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508],
            "x_ub": [2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508],
            "u_lb": [-10.0] * 7,
            "u_ub": [10.0] * 7,
            "q_w": [1e-3] * 7,
            "r_w": [1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 2e-1, 1e-1],
            "qf_w": [1e1] * 7,  # Use custom EE tracking cost instead of State Q
        }
    )

    system_parameters = KinematicDynamicsParameters()
    dynamics = KinematicDynamics(system_parameters)

    # Create Problem
    problem = Problem(problem_parameters, system_parameters)
    problem.dynamics = dynamics

    # 1 Desired End-Effector position and penalty weights

    # x_des = problem_parameters.x_des.detach().clone()

    # x_des = torch.tensor([0.0000, 0.0000, 0.0000, -1.5708, 0.0000, 1.5708, 0.0000])
    # T_ref = torch.tensor(
    #     [
    #         [1.0, 0.0, 0.0, 0.5545],
    #         [0.0, 1.0, 0.0, 0.0000],
    #         [0.0, 0.0, 1.0, 0.7315],
    #         [0.0, 0.0, 0.0, 1.0000],
    #     ]
    # )

    x_des = torch.tensor([0.2800, 0.3000, 0.0000, -1.7900, 0.0000, 2.1200, 0.0000])
    T_ref = torch.tensor(
        [
            [9.6062e-01, 2.7636e-01, 2.8827e-02, 6.0978e-01],
            [2.7623e-01, -9.6106e-01, 8.2894e-03, 1.7534e-01],
            [2.9995e-02, -8.4831e-16, -9.9955e-01, 4.9424e-01],
            [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
        ]
    )

    Q_ee_diag = torch.tensor([5e1, 5e1, 5e1, 5e1, 5e1, 5e1])
    eef_id = model.get_frame_id("fp3_link7")
    ee_cost = EndEffectorTrackingCost(model, data, eef_id, T_ref, Q_ee_diag)

    # Control penalty (velocity minimization)
    Q = problem_parameters.q_w * torch.eye(dynamics.nx).repeat(problem.batch_size, 1, 1)
    R = problem_parameters.r_w * torch.eye(dynamics.nu).repeat(problem.batch_size, 1, 1)
    Qf = problem_parameters.qf_w * torch.eye(dynamics.nx).repeat(
        problem.batch_size, 1, 1
    )
    reg_cost = LqrCost(Q=Q, R=R, x_des=problem_parameters.x_init.detach().clone())
    final_reg_cost = LqrCost(Q=Qf, x_des=x_des)

    if args.load:
        x, u = load_solution(args.load, device=device)
    else:
        x = torch.zeros((problem.batch_size, problem.horizon, problem.n_x))
        for k in range(problem.horizon):
            x[:, k] = problem_parameters.x_init.clone()

        u = torch.zeros((problem.batch_size, problem.horizon - 1, problem.n_u))

    initial_guess = SqpSolution(
        x=x,
        u=u,
        mu=torch.zeros((problem.batch_size, problem.horizon, problem.n_x)),
        nu=torch.zeros((problem.batch_size, problem.horizon - 1, problem.n_h)),
        ksi=[None] * problem.horizon,
    )

    # Randomise Initial state
    initial_guess.x[:, 0] = problem_parameters.x_init.clone()
    noise_std = problem_parameters.noise_std
    noise_dim = len(noise_std)
    initial_guess.x[:, 0] += torch.tensor(noise_std) * torch.randn(
        (batch_size, noise_dim)
    )

    for k in range(problem.horizon - 1):
        problem.costs.append(
            [
                reg_cost,
                ee_cost,
            ]
        )
        problem.constraints[k] = [
            StateBounds(
                problem.n_x,
                problem.n_u,
                problem_parameters.x_lb,
                problem_parameters.x_ub,
            ),
            ControlBounds(
                problem.n_x,
                problem.n_u,
                problem_parameters.u_lb,
                problem_parameters.u_ub,
            ),
        ]
    problem.constraints[-1] = [
        StateBounds(
            problem.n_x, problem.n_u, problem_parameters.x_lb, problem_parameters.x_ub
        )
    ]
    problem.costs.append(
        [
            # final_reg_cost,
            ee_cost,
        ]
    )

    print("Solving End-Effector Tracking Task...")
    solution, log = sqp_solve(problem, sqp_parameters, initial_guess)

    print(log)

    import matplotlib.pyplot as plt
    from diffsqp.utils.plot import plot_trajectories

    print(solution.x[0, -1].tolist())
    plot_trajectories(solution.x, solution.u)
    plt.show()

    if args.save:
        print("Saving solution to ", args.save, "...")
        save_solution(solution, args.save, x_des=x_des)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-batch_size", type=int, help="Batch size", default=1)
    parser.add_argument("-device", type=str, help="Batch size", default="cpu")
    parser.add_argument("-save", type=str, help="Filename to save result")
    parser.add_argument("-load", type=str, help="Filename to load result")
    main(parser.parse_args())
