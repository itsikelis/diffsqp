import time
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
from jax import config

config.update("jax_enable_x64", False)  # don't use double precision

import jax
import jax.numpy as jnp
from jax import jit, vmap

from turbompc.dynamics.integrators import DiscretizationScheme, predict_next_state
from turbompc.dynamics.quadrotor_dynamics import QuadrotorDynamics
from turbompc.problems.optimal_control_problem import OptimalControlProblem
from turbompc.solvers.turbompc_solver import (
    TurboMPCSolver,
    parse_backward_backend,
    parse_forward_backend,
)

"""Utilities"""


def reward(state, control, reference_state, reference_control):
    delta_state = state - reference_state
    delta_control = control - reference_control
    r = -(jnp.sum(delta_state[:3] ** 2) + 1e-1 * jnp.sum(delta_control**2))
    return r


def generate_problem_data(batch_size, noise_std):
    """Generate quadrotor problem data for benchmarking"""

    # Base initial state (x_init)
    x_init = np.array([0.0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    initial_states = np.repeat(x_init[None], repeats=batch_size, axis=0)

    # Randomise Initial state using the provided noise_std scheme
    noise_dim = len(noise_std)
    initial_states[:, :noise_dim] += np.array(noise_std) * np.random.randn(
        batch_size, noise_dim
    )

    # Separately randomise and normalize the quaternion
    initial_states[:, 6:10] = initial_states[:, 6:10] / np.linalg.norm(
        initial_states[:, 6:10], axis=1, keepdims=True
    )

    initial_states = jnp.array(initial_states)

    mass = 0.1 * jnp.ones(1)
    inertia = jnp.array([0.1, 0.01, 0.01, 0.01, 0.1, 0.01, 0.01, 0.01, 0.1])

    return mass, inertia, initial_states


def main(args):
    batch_size = args.batch_size

    horizon = 100
    noise_std = [0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.01, 0.01, 0.01, 0.01, 0.0, 0.0, 0.0]

    # Generate problem data
    mass, inertia, initial_states = generate_problem_data(batch_size, noise_std)
    print(f"device: {inertia.device}")

    turbompc_horizon = horizon - 1  # turbompc horizon is N+1

    # Problem parameters Dictionary
    problem_params = {
        "horizon": turbompc_horizon,
        "discretization_resolution": jnp.array(0.01),
        "discretization_scheme": 0,
        "initial_state": initial_states[0],
        "initial_guess_final_state": jnp.array(
            [1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ),
        "constrain_initial_control": False,
        "initial_control": jnp.array([0.0, 0.0, 0.0, 0.0]),
        "reference_state_trajectory": jnp.repeat(
            jnp.array(
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
            ),
            repeats=turbompc_horizon + 1,
            axis=0,
        ),
        "reference_control_trajectory": jnp.repeat(
            jnp.array([[0.981, 0.0, 0.0, 0.0]]),
            repeats=turbompc_horizon + 1,
            axis=0,
        ),
        "penalize_control_reference": True,
        "weights_penalization_reference_state_trajectory": jnp.array(
            [
                1.0e-4,
                1.0e-4,
                1.0e-4,
                1.0e-1,
                1.0e-1,
                1.0e-1,
                1.0e-4,
                1.0e-4,
                1.0e-4,
                1.0e-4,
                1.0e-1,
                1.0e-1,
                1.0e-1,
            ]
        ),
        "weights_penalization_final_state": jnp.array(
            [
                1.0e5,
                1.0e5,
                1.0e5,
                1.0e3,
                1.0e3,
                1.0e3,
                1.0e3,
                1.0e3,
                1.0e3,
                1.0e3,
                1.0e3,
                1.0e3,
                1.0e3,
            ]
        ),
        "weights_penalization_control_squared": jnp.array(
            [1.0e1, 1.0e-1, 1.0e-1, 1.0e-1]
        ),
        "weights_penalization_control_rate": jnp.array([0.0, 0.0, 0.0, 0.0]),
        "use_slack_variables": jnp.array(False),
        "slack_penalization_weight": jnp.array(0.0),
        "rescale_optimization_variables": False,
        "state_rescaling_min": jnp.array([-1.0] * 13),
        "state_rescaling_max": jnp.array([1.0] * 13),
        "control_rescaling_min": jnp.array([-1.0] * 4),
        "control_rescaling_max": jnp.array([1.0] * 4),
        "control_min_bounds": jnp.array([-2.0, -10.0, -10.0, -10.0]),
        "control_max_bounds": jnp.array([2.0, 10.0, 10.0, 10.0]),
        "dynamics_state_dot_params": {
            "mass": mass,
            "inertia": inertia,
        },
        "mass": mass,
        "inertia": inertia,
    }

    reference_state = problem_params["reference_state_trajectory"][0]
    reference_control = problem_params["reference_control_trajectory"][0]

    # Solver parameters Dictionary
    solver_params = {
        "verbose": False,
        "tol_convergence": 1.0e-4,
        "convergence_criterion": "first_order",
        "num_sqp_iteration_max": 50,
        "linesearch": True,
        "linesearch_eta": 0.4,
        "linesearch_alphas": [
            0.001953125,
            0.00390625,
            0.0078125,
            0.015625,
            0.03125,
            0.0625,
            0.125,
            0.25,
            0.5,
            1.0,
        ],
        "admm": {
            "rho": 0.1,
            "sigma": 1.0e-6,
            "max_iter": 50,
            "eps_abs": 1.0e-3,
            "eps_rel": 1.0e-4,
            "rho_min": 1.0e-6,
            "rho_max": 1.0e6,
            "check_termination_every": 1,
            "adapt_rho_every": 25,
            "adaptive_rho_tolerance": 5,
            "relaxation_parameter": 1.6,  # in (1, 2)
            "rho_f_factor": 1000.0,
            "pcg": {
                "max_iter": 10000,
                "tol_epsilon": 1.0e-20,
            },
        },
    }

    fwd_backend = parse_forward_backend("admm_fused_cudss")
    bwd_backend = parse_backward_backend("direct_cudss_ffi")

    dynamics = QuadrotorDynamics()
    problem = OptimalControlProblem(dynamics=dynamics, params=problem_params)
    solver = TurboMPCSolver(
        program=problem,
        params=solver_params,
        forward_backend=fwd_backend,
        backward_backend=bwd_backend,
    )

    def solver_initial_guess(initial_state):
        params = {**problem_params, "initial_state": initial_state}
        # straight line
        guess = solver.initial_guess(params)
        # normalize quaternion
        quaternions = guess.states[:, 6:10]

        def normalize(q):
            return q / jnp.linalg.norm(q)

        quaternions = vmap(normalize)(quaternions)
        #
        controls = jnp.repeat(
            jnp.array([[0.1 * 9.81, 0.0, 0.0, 0.0]]), repeats=horizon, axis=0
        )
        guess = guess._replace(
            states=guess.states.at[:, 6:10].set(quaternions), controls=controls
        )
        return guess

    # trainable parameters
    weights = {
        k: problem_params[k]
        for k in [
            "weights_penalization_reference_state_trajectory",
            "weights_penalization_control_squared",
            "weights_penalization_final_state",
        ]
    }

    ### Solve the problem with nominal MPC parameters
    start = time.time()
    solution = solver.solve(
        solver.initial_guess(problem_params), problem_params, weights
    )
    solve_time = time.time() - start

    constr_viol_inf = jnp.maximum(
        jnp.max(solution.solver_stats.eq_constraints_violations),
        jnp.max(solution.solver_stats.ineq_constraints_violations),
    )

    log = {
        "final_state": solution.states[-1].tolist(),
        "sqp_iterations": solution.num_iter.tolist(),
        "admm_iterations": solution.admm_iters.tolist(),
        "solve_time": solve_time,
        "constraint_violation": constr_viol_inf.tolist(),
    }

    if args.save:
        print(f"Saving solution to {args.save}.pt...")
        with open(args.save + ".json", "w", encoding="utf-8") as f:
            json.dump(log, f, indent=4)

    # control_min_bounds = jnp.asarray(problem_params["control_min_bounds"])
    # control_max_bounds = jnp.asarray(problem_params["control_max_bounds"])
    #
    # # plot rollouts with base weights
    # fig, axes = plt.subplots(
    #     nrows=dynamics.num_states + dynamics.num_controls, figsize=(8, 12)
    # )
    # for i, ax in enumerate(axes.flatten()):
    #     if i < dynamics.num_states:
    #         ax.plot(solution.states[:, i], "b--")
    #         ax.set_ylabel(dynamics.names_states[i])
    #     else:
    #         control_idx = i - dynamics.num_states
    #         ax.plot(solution.controls[:, control_idx], "b--")
    #         # ax.axhline(
    #         #     control_min_bounds[control_idx],
    #         #     color="r",
    #         #     linestyle="--",
    #         #     linewidth=1.0,
    #         #     label="Bounds",
    #         # )
    #         # ax.axhline(
    #         #     control_max_bounds[control_idx], color="r", linestyle="--", linewidth=1.0
    #         # )
    #         ax.set_ylabel(dynamics.names_controls[control_idx])
    #     ax.grid(True)
    #     ax.legend()
    #
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-batch_size", type=int, help="Batch size", default=1)
    parser.add_argument("-device", type=str, help="Batch size", default="cuda")
    parser.add_argument("-save", type=str, help="Filename to save result")
    parser.add_argument("-load", type=str, help="Filename to load result")
    main(parser.parse_args())
