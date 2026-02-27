import time
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
from jax import config

config.update("jax_enable_x64", True)  # use double precision

import jax
import jax.numpy as jnp
from jax import jit, vmap

from diffmpc.dynamics.integrators import DiscretizationScheme, predict_next_state
from diffmpc.dynamics.cartpole_dynamics import CartpoleDynamics
from diffmpc.problems.optimal_control_problem import OptimalControlProblem
from diffmpc.solvers.sqp import SQPSolver
from diffmpc.utils.load_params import (
    load_problem_params,
    load_solver_params,
)


def generate_problem_data(params, n_batch, std, seed):
    """Generate initial states for benchmarking"""
    key = jax.random.PRNGKey(seed)
    initial_states = params["initial_state"] + std * jax.random.normal(
        key, (n_batch, 4)
    )
    return initial_states


def main(args):
    n_batch = args.nb  # Adjust based on your GPU memory
    std = args.std

    # load_trajectory(problem_params)
    problem_params = load_problem_params("cartpole.yaml")
    initial_states = generate_problem_data(
        problem_params, n_batch, std, seed=int(time.time())
    )

    dynamics = CartpoleDynamics()
    problem = OptimalControlProblem(dynamics=dynamics, params=problem_params)

    # Load solver
    solver_params = load_solver_params("sqp.yaml")
    solver_params["tol_convergence"] = 1.0e-4
    solver_params["num_scp_iteration_max"] = 100
    solver_params["pcg"]["tol_epsilon"] = 1.0e-6
    solver_params["linesearch"] = True
    solver_params["warm_start_backward"] = False
    solver_params["linesearch_alphas"] = [1.0, 0.5, 0.25, 0.125, 0.0625]
    solver = SQPSolver(program=problem, params=solver_params)

    def solve_single_instance(init_state):
        local_params = problem_params.copy()
        local_params["initial_state"] = init_state
        init_guess = solver.initial_guess(local_params)
        return solver.solve(init_guess, local_params, weights)

    weights = {
        k: problem_params[k]
        for k in [
            "weights_penalization_reference_state_trajectory",
            "weights_penalization_control_squared",
        ]
    }

    parallel_solver = jit(vmap(solve_single_instance))

    # print(f"Starting parallel solve for {n_batch} instances...")

    # Warm-up (JIT compilation happens here)
    _ = parallel_solver(initial_states[:1])

    start = time.time()
    solutions = parallel_solver(initial_states)
    # Ensure GPU finishes before stopping timer
    jax.block_until_ready(solutions)
    end = time.time()

    log = {
        "t_solve_s": end - start,
        "n_batch": n_batch,
        "num_iter": solutions.num_iter.tolist(),
        "status": solutions.status.tolist(),
        "convergence_error": solutions.convergence_error.tolist(),
        "err_final_state": jnp.abs(
            jnp.square(solutions.states[:, -1] - problem_params["final_state"])
        ).tolist(),
    }

    with open("log.json", "w") as f:
        json.dump(log, f, indent=4)

    # print(f"Total time for {n_batch} problems: {end - start:.4f}s")
    # print(f"Average time per problem: {(end - start)/n_batch:.4f}s")

    # import matplotlib.pyplot as plt
    # print(solution.states)
    # plt.plot(solutions.states[0, :, 0])
    # plt.plot(solutions.states[0, :, 2])
    # plt.plot(solution.states[:, 4])
    # plt.plot(solution.controls[:, 2])
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-nb", type=int, help="Batch size")
    parser.add_argument(
        "-std", type=float, help="Initial state noise standard deviation"
    )
    main(parser.parse_args())
