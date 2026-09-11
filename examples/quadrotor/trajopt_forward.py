import argparse
import torch

from diffsqp.problems import Problem, ProblemParameters
from diffsqp.costs import QuadrotorTrackingCost, LqrCost
from diffsqp.solvers import sqp_solve, SqpParameters
from diffsqp.dynamics import QuadrotorDynamics, QuadrotorParameters
from diffsqp.constraints import StateBounds, ControlBounds
from diffsqp.types import SqpSolution

from diffsqp.utils.load_save import *


def main(args):
    device = args.device
    torch.set_default_device(device)
    batch_size = args.batch_size

    sqp_parameters = SqpParameters(
        **{
            ## ADMM ##
            "admm_max_iter": 30,
            "admm_alpha": 1.6,
            "admm_sigma": 1e-6,
            # Rho related
            "admm_reset_rho": False,
            "admm_update_rho": False,
            "admm_rho_init": 0.4,
            "admm_rho_min": 1e-6,
            "admm_rho_max": 1e8,
            "admm_adaptive_rho_tolerance": 10.0,
            "admm_rho_update_iter_freq": 10,
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
            "sqp_max_iter": 100,
            "lqr_reg_init": 1e-5,
            "merit_mu": 1e6,
            "armijo_beta": 1e-4,
            "ls_max_iter": 10,
            "sqp_cost_eps": 1e-2,
            "sqp_viol_eps": 1e-3,
            "qp_solver": "lqr",
            "ls_function": "merit",
        }
    )

    problem_parameters = ProblemParameters(
        **{
            "inverse_dynamics": False,
            "n_h": 0,
            "batch_size": batch_size,
            "dt": 0.01,
            "tf": 1.0,
            "x_init": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "x_des": [1.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "noise_std": [0.01, 0.01, 0.01],
            # State-control bounds
            "x_lb": [
                -100.0,
                -100.0,
                -100.0,
                -1e6,
                -1e6,
                -1e6,
                -1e6,
                -1e6,
                -1e6,
                -1e6,
                -1e6,
                -1e6,
                -1e6,
            ],
            "x_ub": [
                100.0,
                100.0,
                100.0,
                1e6,
                1e6,
                1e6,
                1e6,
                1e6,
                1e6,
                1e6,
                1e6,
                1e6,
                1e6,
            ],
            "u_lb": [-1e6, -1e6, -1e6, -1e6],
            "u_ub": [1e6, 1e6, 1e6, 1e6],
            # Cost weights
            "q_w": [
                1e-6,
                1e-6,
                1e-6,
                1e-8,
                1e-8,
                1e-8,
                1e-8,
                1e-6,
                1e-6,
                1e-6,
                1e-6,
                1e-6,
            ],
            "r_w": [1e-1, 1e-1, 1e-1, 1e-1],
            "qf_w": [1e5, 1e5, 1e5, 1e5, 1e5, 1e5, 1e5, 1e5, 1e5, 1e5, 1e5, 1e5],
        }
    )

    system_parameters = QuadrotorParameters(
        **{
            "name": "quadrotor",
            "n_x": 13,  # Number of state elements
            "n_q": 7,  # Number of position elements in state
            "n_v": 6,  # Number of velocity elements in state
            "n_j": 0,  # Number of joints
            "n_u": 4,  # Number of DoF
            "mass": 0.1,
            "inertia": [0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1],
            "grav": 9.81,
        }
    )

    # print(sqp_parameters)
    # print(problem_parameters)
    # print(system_parameters)

    dynamics = QuadrotorDynamics(system_parameters)

    # Create problem
    problem = Problem(problem_parameters, system_parameters)

    # Costs
    Q = torch.zeros(dynamics.nx, dynamics.nx).repeat(
        problem_parameters.batch_size, 1, 1
    )
    R = problem_parameters.r_w * torch.eye(dynamics.nu).repeat(
        problem_parameters.batch_size, 1, 1
    )

    initial_guess = SqpSolution(
        x=torch.zeros((problem.batch_size, problem.horizon, problem.n_x)),
        u=torch.zeros((problem.batch_size, problem.horizon - 1, problem.n_u)),
        mu=torch.zeros((problem.batch_size, problem.horizon, problem.n_x)),
        nu=torch.zeros((problem.batch_size, problem.horizon - 1, problem.n_h)),
        ksi=[None] * problem.horizon,
    )

    if args.load:
        x, u = load_solution(args.load, device=device)
        intial_guess.x = x
        intial_guess.u = u
    else:
        x_init_tensor = problem_parameters.x_init.detach().clone()
        x_des_tensor = problem_parameters.x_des.detach().clone()

        initial_guess.x[:] = x_init_tensor
        alphas = torch.linspace(
            0, 1, problem.horizon, device=x_init_tensor.device
        ).unsqueeze(1)
        pos_traj = (1.0 - alphas) * x_init_tensor[0:3] + alphas * x_des_tensor[0:3]
        initial_guess.x[:, :, 0:3] = pos_traj
        initial_guess.x[:, -1] = x_des_tensor

        # Initialize u
        m = system_parameters.mass
        grav = system_parameters.grav
        initial_guess.u = torch.Tensor([m * grav, 0.0, 0.0, 0.0]).repeat(
            problem.batch_size, problem.horizon - 1, 1
        )

    # Randomise Initial state
    noise_std = problem_parameters.noise_std
    noise_dim = len(noise_std)
    initial_guess.x[:, 0] = problem_parameters.x_init.clone()
    initial_guess.x[:, 0, :noise_dim] += torch.tensor(noise_std) * torch.randn(
        (batch_size, noise_dim)
    )

    # Set stage costs, constraints and initial guess
    for k in range(problem.horizon - 1):
        problem.costs.append(
            [
                LqrCost(Q=Q, R=R),
                QuadrotorTrackingCost(
                    Q_diag=problem_parameters.q_w,
                    x_des=problem_parameters.x_init.detach().clone(),
                ),
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
    # Terminal stage
    problem.costs.append(
        [
            QuadrotorTrackingCost(
                Q_diag=problem_parameters.qf_w,
                x_des=problem_parameters.x_des.detach().clone(),
            )
        ]
    )
    problem.constraints[-1] = [
        StateBounds(
            problem.n_x,
            problem.n_u,
            problem_parameters.x_lb,
            problem_parameters.x_ub,
        )
    ]

    # Dynamics Constraints
    problem.dynamics = dynamics

    # Solve
    print("Solving Quadrotor Trajopt Task...")
    solution, log = sqp_solve(problem, sqp_parameters, initial_guess)

    print(log)

    if args.save:
        print(f"Saving solution to {args.save}.pt...")
        save_solution(solution, args.save, x_des=problem_parameters.x_des)
        log.save_to_json(args.save)

    # import matplotlib.pyplot as plt
    # from diffsqp.utils.plot import plot_trajectories
    #
    # plot_trajectories(solution.x, solution.u)
    # plt.show()

    # Animate:
    # from diffsqp.utils.animate import QuadrotorAnimator
    #
    # animator = QuadrotorAnimator(
    #     solution.x,
    #     problem_parameters.dt,
    #     problem_parameters.batch_size,
    # )
    # animator.animate(step_size=2)
    # animator.save(filename="quadrotor.mp4", step_size=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-batch_size", type=int, help="Batch size", default=1)
    parser.add_argument("-device", type=str, help="Batch size", default="cpu")
    parser.add_argument("-save", type=str, help="Filename to save result")
    parser.add_argument("-load", type=str, help="Filename to load result")
    main(parser.parse_args())
