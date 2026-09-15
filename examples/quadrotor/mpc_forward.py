import argparse
import torch
import matplotlib.pyplot as plt

from diffsqp.problems import Problem, ProblemParameters
from diffsqp.costs import QuadrotorTrackingCost, LqrCost
from diffsqp.solvers import sqp_solve, SqpParameters
from diffsqp.dynamics import QuadrotorDynamics, QuadrotorParameters
from diffsqp.constraints import StateBounds, ControlBounds
from diffsqp.types import SqpSolution
from diffsqp.utils.load_save import *
from diffsqp.utils.plot import plot_trajectories
from diffsqp.utils.animate import QuadrotorAnimator


def main(args):
    device = args.device
    torch.set_default_device(device)
    batch_size = args.batch_size
    dt_mpc = args.dt_mpc
    N_mpc = args.n_mpc

    sqp_parameters = SqpParameters(
        **{
            ## ADMM ##
            "admm_max_iter": 30,
            "admm_alpha": 1.6,
            "admm_sigma": 1e-6,
            "admm_reset_rho": False,
            "admm_update_rho": False,
            "admm_rho_init": 0.4,
            "admm_rho_min": 1e-6,
            "admm_rho_max": 1e8,
            "admm_adaptive_rho_tolerance": 10.0,
            "admm_rho_update_iter_freq": 10,
            "admm_warm_start_unconstrained": True,
            "admm_reset_ksi": False,
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
            "dt": 0.01,  # Solver dt (different from dt_mpc)
            "tf": 1.0,
            "x_init": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "x_des": [1.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "noise_std": [0.01, 0.01, 0.01, 0.001, 0.001, 0.001],
            "x_lb": [-100.0] * 3 + [-1e6] * 10,
            "x_ub": [100.0] * 3 + [1e6] * 10,
            "u_lb": [-1e6] * 4,
            "u_ub": [1e6] * 4,
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
            "qf_w": [1e5] * 12,
        }
    )

    system_parameters = QuadrotorParameters(
        **{
            "name": "quadrotor",
            "n_x": 13,
            "n_q": 7,
            "n_v": 6,
            "n_j": 0,
            "n_u": 4,
            "mass": 0.1,
            "inertia": [0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1],
            "grav": 9.81,
        }
    )

    dynamics = QuadrotorDynamics(system_parameters)
    problem = Problem(problem_parameters, system_parameters)
    problem.dynamics = dynamics

    Q = torch.zeros(dynamics.nx, dynamics.nx).repeat(
        problem_parameters.batch_size, 1, 1
    )
    R = problem_parameters.r_w * torch.eye(dynamics.nu).repeat(
        problem_parameters.batch_size, 1, 1
    )

    # Costs and constraints mapping
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
            problem.n_x, problem.n_u, problem_parameters.x_lb, problem_parameters.x_ub
        )
    ]

    # --- 0) Initialization ---
    initial_guess = SqpSolution(
        x=torch.zeros((problem.batch_size, problem.horizon, problem.n_x)),
        u=torch.zeros((problem.batch_size, problem.horizon - 1, problem.n_u)),
        mu=torch.zeros((problem.batch_size, problem.horizon, problem.n_x)),
        nu=torch.zeros((problem.batch_size, problem.horizon - 1, problem.n_h)),
        ksi=[None] * problem.horizon,
    )

    x_init_tensor = problem_parameters.x_init.detach().clone()
    x_des_tensor = problem_parameters.x_des.detach().clone()

    initial_guess.x[:] = x_init_tensor
    alphas = torch.linspace(
        0, 1, problem.horizon, device=x_init_tensor.device
    ).unsqueeze(1)
    initial_guess.x[:, :, 0:3] = (1.0 - alphas) * x_init_tensor[
        0:3
    ] + alphas * x_des_tensor[0:3]
    initial_guess.x[:, -1] = x_des_tensor

    m, grav = system_parameters.mass, system_parameters.grav
    initial_guess.u = torch.tensor([m * grav, 0.0, 0.0, 0.0]).repeat(
        problem.batch_size, problem.horizon - 1, 1
    )

    print("Solving Initial Warm-Start Batch...")
    solution, log = sqp_solve(problem, sqp_parameters, initial_guess)

    # --- Robust MPC Setup ---
    true_state = problem_parameters.x_init.clone().unsqueeze(0)  # (1, n_x)

    noise_std = torch.tensor(problem_parameters.noise_std, device=device)
    noise_dim = len(noise_std)

    # 1. Add noise to the true initial state
    true_noise = torch.randn((1, noise_dim), device=device) * noise_std
    true_state[:, :noise_dim] += true_noise

    # 2. Post-normalize the quaternion of the true state (indices 3 to 6)
    true_quat = true_state[:, 3:7]
    true_state[:, 3:7] = true_quat / torch.norm(true_quat, p=2, dim=1, keepdim=True)

    sqp_parameters.ls_max_iter = 0
    sqp_parameters.admm_max_iter = 5
    sqp_parameters.sqp_max_iter = 1

    # --- 1) MPC Loop ---
    print(f"Starting MPC loop ({N_mpc} iterations, dt_mpc={dt_mpc})...")
    for mpc_step in range(N_mpc):
        # 1a. Randomize the batched initial states around the true quadrotor state
        current_state_batch = true_state.repeat(batch_size, 1)
        noise = torch.randn((batch_size, noise_dim), device=device) * noise_std
        current_state_batch[:, :noise_dim] += noise

        # Enforce current position as the start of the horizon
        solution.x[:, 0, :] = current_state_batch

        # 1b. Solve the problem from this new starting point
        solution, log = sqp_solve(problem, sqp_parameters, solution)

        # 1c. Apply the mean action and roll out dynamics by dt_mpc
        u0_mean = solution.u[:, 0, :].mean(dim=0, keepdim=True)  # (1, n_u)
        applied_controls.append(u0_mean.squeeze().cpu())

        # Integrate true dynamics forward using standard explicit Euler step
        true_state = dynamics.f(true_state, u0_mean, dt_mpc)
        true_trajectory.append(true_state.squeeze().cpu())

        # 1d. Shift horizon based on accumulated time
        accumulated_time += dt_mpc

        # Calculate how many full solver steps have passed (adding 1e-6 for float precision)
        shifts = int((accumulated_time + 1e-6) // dt_solver)

        if shifts > 0 and shifts < problem.horizon:
            # Shift state horizon
            solution.x[:, :-shifts, :] = solution.x[:, shifts:, :].clone()
            # Pad the end with the last known state
            solution.x[:, -shifts:, :] = solution.x[:, -1:, :].clone()

            # Shift control horizon
            solution.u[:, :-shifts, :] = solution.u[:, shifts:, :].clone()
            solution.u[:, -shifts:, :] = solution.u[:, -1:, :].clone()

            # Shift dual variables
            solution.mu[:, :-shifts, :] = solution.mu[:, shifts:, :].clone()
            solution.mu[:, -shifts:, :] = solution.mu[:, -1:, :].clone()

            # Decrement the time we just accounted for
            accumulated_time -= shifts * dt_solver
        print(f"MPC Step {mpc_step + 1}/{N_mpc} | Pos: {true_state[0, :3].tolist()}")

    # --- 2) Save and Plot Results ---
    mpc_x = torch.stack(true_trajectory).unsqueeze(0)  # (1, N_mpc+1, n_x)
    mpc_u = torch.stack(applied_controls).unsqueeze(0)  # (1, N_mpc, n_u)

    if args.save:
        save_dict = {"x": mpc_x, "u": mpc_u, "dt_mpc": dt_mpc}
        torch.save(save_dict, f"{args.save}_mpc.pt")
        print(f"MPC trajectory saved to {args.save}_mpc.pt")

    # print("Plotting MPC Trajectory...")
    # plot_trajectories(mpc_x, mpc_u)
    # plt.show()

    # print("Animating MPC Trajectory...")
    # animator = QuadrotorAnimator(mpc_x, dt_mpc, 1)
    # animator.animate(step_size=1)
    # animator.save("robust_mpc.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-batch_size", type=int, default=16, help="Batch size for robust MPC"
    )
    parser.add_argument("-device", type=str, default="cpu")
    parser.add_argument(
        "-save", type=str, default="mpc_result", help="Filename prefix to save result"
    )
    parser.add_argument("-load", type=str, help="Filename to load result")
    parser.add_argument(
        "-dt_mpc", type=float, default=0.001, help="Execution time step for MPC"
    )
    parser.add_argument(
        "-n_mpc", type=int, default=1000, help="Number of MPC iterations to run"
    )
    main(parser.parse_args())
