import argparse
import torch
import bard
from dataclasses import dataclass
import matplotlib.pyplot as plt

from diffsqp.problems import Problem, ProblemParameters
from diffsqp.costs import LqrCost, Cost
from diffsqp.solvers import sqp_solve, SqpParameters
from diffsqp.dynamics.base_dynamics import Dynamics
from diffsqp.constraints import StateBounds, ControlBounds
from diffsqp.types import SqpSolution

from diffsqp.utils.plot import plot_trajectories
from diffsqp.utils.load_save import *
import matplotlib.pyplot as plt

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


#####################################
# Custom end-effector tracking cost #
#####################################


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

        # 1. Position Error (3D)
        p_ee = transforms[..., :3, 3]
        p_ref = T_ref[..., :3, 3]
        pos_err = p_ee - p_ref

        # 2. Orientation Error (3D) using SO(3) skew-symmetric mapping
        R_ee = transforms[..., :3, :3]
        R_ref = T_ref[..., :3, :3]

        # e_R = 0.5 * vee(R_ee * R_ref^T - R_ref * R_ee^T)
        R_ref_T = R_ref.transpose(-1, -2)
        R_ee_T = R_ee.transpose(-1, -2)
        S = torch.matmul(R_ee, R_ref_T) - torch.matmul(R_ref, R_ee_T)

        # Extract the vector part (vee operator) from the skew-symmetric matrix S
        ori_err = 0.5 * torch.stack([S[..., 2, 1], S[..., 0, 2], S[..., 1, 0]], dim=-1)

        # Combine into a 6D error vector
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
    device = "cuda"
    torch.set_default_device("cuda")
    batch_size = args.b
    model = bard.build_model_from_urdf("fp3.urdf", floating_base=False)
    model.to(dtype=torch.float32, device=device)
    data = bard.create_data(model, max_batch_size=batch_size)
    eef_id = model.get_frame_id("fp3_link7")

    sqp_parameters = SqpParameters(
        **{
            "admm_max_iter": 50,
            "admm_alpha": 1.6,
            "admm_sigma": 1e-6,
            "admm_reset_rho": False,
            "admm_update_rho": False,
            "admm_rho_init": 0.8,
            "admm_rho_min": 1e-6,
            "admm_rho_max": 1e3,
            "admm_adaptive_rho_tolerance": 10.0,
            "admm_rho_update_iter_freq": 10,
            "admm_warm_start_unconstrained": False,
            "admm_reset_ksi": False,
            "admm_abs_tolerance": 0.01,
            "admm_abs_tolerance_final": -1.0,
            "admm_rel_tolerance": 0.0001,
            "admm_rel_tolerance_final": -1.0,
            "admm_tolerance_update_steps": 0,
            "sqp_max_iter": 100,
            "merit_mu": 1e6,
            "armijo_beta": 1e-4,
            "ls_max_iter": 10,
            "sqp_eps": 1e-4,
            "qp_solver": "lqr",
            "ls_function": "filter",
        }
    )

    problem_parameters = ProblemParameters(
        **{
            "inverse_dynamics": False,
            "n_h": 0,
            "batch_size": batch_size,
            "tf": 1.0,
            "dt": 0.01,
            "x_init": [0.0, 0.0, 0.0, -torch.pi / 2.0, 0.0, torch.pi / 2.0, 0.0],
            "x_des": [0.0, 0.0, 0.2, -torch.pi / 2.0, 0.0, torch.pi / 2.0, 0.0],
            "noise_std": 0.0,
            "x_lb": [-1e6] * 7,
            "x_ub": [1e6] * 7,
            "u_lb": [-1e6] * 7,
            "u_ub": [1e6] * 7,
            "q_w": [1e-5] * 7,
            "r_w": [1e-1] * 7,
            "qf_w": [1e-5] * 7,  # Use custom EE tracking cost instead of State Q
        }
    )

    system_parameters = KinematicDynamicsParameters()
    dynamics = KinematicDynamics(system_parameters)

    # --- 3. Build & Solve Problem ---

    problem = Problem(problem_parameters, system_parameters)
    problem.dynamics = dynamics

    # 1 Desired End-Effector position and penalty weights

    T_ref = torch.tensor(
        [
            [9.9500e-01, -4.0467e-16, 9.9833e-02, 6.9161e-01],
            [-4.7893e-16, -1.0000e00, 7.1989e-16, -6.7600e-17],
            [9.9833e-02, -7.6411e-16, -9.9500e-01, 4.5008e-01],
            [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
        ]
    )

    # T_ref = torch.tensor(
    #     [
    #         [
    #             [9.9500e-01, -4.0467e-16, 9.9833e-02, 6.9161e-01],
    #             [-4.7893e-16, -1.0000e00, 7.1989e-16, -6.7600e-17],
    #             [9.9833e-02, -7.6411e-16, -9.9500e-01, 4.5008e-01],
    #             [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
    #         ],
    #         [
    #             [9.8007e-01, 1.5816e-01, -1.2023e-01, 5.4345e-01],
    #             [1.9867e-01, -7.8022e-01, 5.9312e-01, 1.1016e-01],
    #             [-2.6876e-16, -6.0519e-01, -7.9608e-01, 7.3150e-01],
    #             [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
    #         ],
    #     ]
    # )
    Q_ee_diag = torch.tensor([1e3, 1e3, 1e3, 1e2, 1e2, 1e2])
    ee_cost = EndEffectorTrackingCost(model, data, eef_id, T_ref, Q_ee_diag)

    # Control penalty (velocity minimization)
    Q = problem_parameters.q_w * torch.eye(dynamics.nx).repeat(problem.batch_size, 1, 1)
    R = problem_parameters.r_w * torch.eye(dynamics.nu).repeat(problem.batch_size, 1, 1)
    Qf = problem_parameters.qf_w * torch.eye(dynamics.nx).repeat(
        problem.batch_size, 1, 1
    )
    reg_cost = LqrCost(Q=Q, R=R, x_des=problem_parameters.x_init.detach().clone())
    final_reg_cost = LqrCost(Q=Qf, x_des=problem_parameters.x_des.detach().clone())

    if args.load:
        x, u = load_solution(args.load, device=device)
        # u = torch.zeros((problem.batch_size, problem.horizon - 1, problem.n_u))
    else:
        x = problem_parameters.x_init.clone().expand(
            batch_size, problem.horizon, problem.n_x
        )
        u = torch.zeros((problem.batch_size, problem.horizon - 1, problem.n_u))

    initial_guess = SqpSolution(
        x=torch.zeros((problem.batch_size, problem.horizon, problem.n_x)),
        u=torch.zeros((problem.batch_size, problem.horizon - 1, problem.n_u)),
        mu=torch.zeros((problem.batch_size, problem.horizon, problem.n_x)),
        nu=torch.zeros((problem.batch_size, problem.horizon - 1, problem.n_h)),
        ksi=[None] * problem.horizon,
    )

    initial_guess.x[:, 0] = problem_parameters.x_init.clone()
    for k in range(problem.horizon - 1):
        problem.costs.append(
            [
                # ee_cost,
                reg_cost,
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

    # initial_guess.x[:, -1] = problem_parameters.x_init.clone()
    problem.constraints[-1] = [
        StateBounds(
            problem.n_x, problem.n_u, problem_parameters.x_lb, problem_parameters.x_ub
        )
    ]
    problem.costs.append(
        [
            ee_cost,
            # final_reg_cost,
        ]
    )

    print("Solving End-Effector Tracking Task...")
    solution, log = sqp_solve(problem, sqp_parameters, initial_guess)

    print(log)
    print(f"Time elapsed: {log.solve_wall_time_s} s.")

    plot_trajectories(solution.x, solution.u)

    if args.save:
        print("Saving")
        save_solution(solution, args.save)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", type=int, help="Batch size", default=1)
    parser.add_argument("-save", type=str, help="Filename to save result")
    parser.add_argument("-load", type=str, help="Filename to load result")
    main(parser.parse_args())
