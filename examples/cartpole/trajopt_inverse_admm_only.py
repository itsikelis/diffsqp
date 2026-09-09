import argparse
import torch

from diffsqp.problems import Problem, ProblemParameters
from diffsqp.costs import LqrCost
from diffsqp.solvers import sqp_solve, SqpParameters
from diffsqp.dynamics import Dynamics
from diffsqp.dynamics import CartPoleParameters
from diffsqp.constraints import (
    CartPoleUnderactuation,
    StateBounds,
    ControlBounds,
)
from diffsqp.types import SqpSolution


def main(args):
    device = args.device
    torch.set_default_device(device)
    batch_size = args.batch_size

    sqp_parameters = SqpParameters(
        **{
            ## ADMM ##
            "admm_max_iter": 250,
            "admm_alpha": 1.6,
            "admm_sigma": 1e-6,
            # Rho related
            "admm_reset_rho": True,
            "admm_update_rho": True,
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
            "merit_mu": 1e5,
            "armijo_beta": 1e-3,
            "ls_max_iter": 10,
            "sqp_eps": 1e-4,
            "qp_solver": "lqr",
            "ls_function": "merit",
        }
    )

    problem_parameters = ProblemParameters(
        **{
            "inverse_dynamics": False,
            "n_h": 0,
            "batch_size": 1,
            "dt": 0.01,
            "tf": 1.0,
            "x_init": [0.0, 0.0, 0.0, 0.0],
            "x_des": [0.0, 3.14159, 0.0, 0.0],
            "noise_std": [0.0, 0.0, 0.0, 0.0],
            # State-control bounds
            "x_lb": [-2.0, -1e6, -5.0, -15.0],
            "x_ub": [2.0, 1e6, 5.0, 15.0],
            "u_lb": [-1e6, -1e6],
            "u_ub": [1e6, 1e6],
            # Cost weights
            "q_w": [1e-6, 1e-6, 1e-6, 1e-6],
            "r_w": [1e-3, 1e-3],
            "qf_w": [1e5, 1e5, 1e5, 1e5],
        }
    )

    system_parameters = CartPoleParameters(
        **{
            "name": "cartpole",
            "n_x": 4,
            "n_q": 2,
            "n_v": 2,
            "n_j": 2,
            "n_u": 2,
            "mc": 0.5,
            "mp": 0.3,
            "lp": 0.2,
            "grav": 9.81,
        }
    )

    print(sqp_parameters)
    print(problem_parameters)
    print(system_parameters)

    dynamics = Dynamics(
        nx=system_parameters.n_x,
        nu=system_parameters.n_u,
        nq=system_parameters.n_q,
        nv=system_parameters.n_v,
    )
    underactuation = CartPoleUnderactuation(system_parameters)

    # Create problem
    problem = Problem(problem_parameters, system_parameters)

    # Costs
    Q = problem_parameters.q_w * torch.eye(dynamics.nx).repeat(
        problem_parameters.batch_size, 1, 1
    )
    R = problem_parameters.r_w * torch.eye(dynamics.nu).repeat(
        problem_parameters.batch_size, 1, 1
    )
    Qf = problem_parameters.qf_w * torch.eye(dynamics.nx).repeat(
        problem_parameters.batch_size, 1, 1
    )

    if args.load:
        x, u = load_solution(args.load, device=device)
    else:
        x = torch.zeros((problem.batch_size, problem.horizon, problem.n_x))
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

    # Set stage costs, constraints and initial guess
    for k in range(problem.horizon - 1):
        problem.costs.append([LqrCost(Q=Q, R=R)])
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
            CartPoleUnderactuation(system_parameters),
        ]
    # Terminal stage
    problem.costs.append(
        [LqrCost(Q=Qf, x_des=problem_parameters.x_des.detach().clone())]
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
    print("Solving Cartpole Swingup Task...")
    solution, log = sqp_solve(problem, sqp_parameters, initial_guess)

    print(log)

    if args.save:
        print("Saving solution to ", args.save, "...")
        save_solution(solution, args.save, x_des=problem_parameters.x_des)

    # import matplotlib.pyplot as plt
    # from diffsqp.utils.plot import plot_trajectories
    #
    # plot_trajectories(solution.x, solution.u)
    # plt.show()

    # # Animate:
    # from diffsqp.utils.animate import CartPoleAnimator
    #
    # animator = CartPoleAnimator(
    #     solution.x,
    #     system_parameters.lp,
    #     problem_parameters.dt,
    #     problem_parameters.batch_size,
    # )
    # animator.animate(step_size=2)
    # # animator.save(filename="admm.mp4", step_size=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-batch_size", type=int, help="Batch size", default=1)
    parser.add_argument("-device", type=str, help="Batch size", default="cpu")
    parser.add_argument("-save", type=str, help="Filename to save result")
    parser.add_argument("-load", type=str, help="Filename to load result")
    main(parser.parse_args())
