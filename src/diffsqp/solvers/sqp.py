import sys
import time
import torch
from copy import copy
from diffsqp.utils.math import mm, mv, inf_norm
from typing import List

from diffsqp.problems import Problem, ProblemParameters
from diffsqp.solvers import QP
from diffsqp.solvers import admm_qp_solve, lqr_solve
from dataclasses import dataclass
from diffsqp.types import SqpSolution, AdmmSolution, LqrSolution


class SqpParameters:
    def __init__(self, **args):
        self.admm_max_iter: int = args["admm_max_iter"]
        self.admm_eps: float = args["admm_eps"]
        self.admm_alpha: float = args["admm_alpha"]
        self.admm_sigma: float = args["admm_sigma"]
        self.admm_rho: float = args["admm_rho"]
        self.admm_warm_start: float = args["admm_warm_start"]
        self.admm_initialize_unconstrained: float = args[
            "admm_initialize_unconstrained"
        ]

        self.sqp_max_iter: int = args["sqp_max_iter"]
        self.merit_mu: float = args["merit_mu"]
        self.ls_max_iter: int = args["ls_max_iter"]
        self.sqp_eps: float = args["sqp_eps"]
        self.qp_solver: str = args["qp_solver"]
        self.ls_function: str = args["ls_function"]

    def __str__(self) -> str:
        return (
            f"=== SQP Parameters ===\n"
            f" QP Solver       : {self.qp_solver}\n"
            f" Line Search Fn  : {self.ls_function}\n"
            f" SQP Max Iter    : {self.sqp_max_iter}\n"
            f" Line Search Max : {self.ls_max_iter}\n"
            f" SQP Tolerance   : {self.sqp_eps:.2e}\n"
            f"======================"
        )


class SqpSolutionLog:
    def __init__(self):
        self.envs_terminated: int = 0

        self.total_cost: List[float] = []
        self.constraint_violation: List[float] = []

        self.solve_wall_time_s: int = 0
        self.sqp_iterations: int = 0

        self.termination_time_s: float = 0.0
        self.admm_iters: List[float] = []
        self.ls_iters: List[float] = []
        self.ls_alphas: List[float] = []

        # GPU related
        self.cuda_reserved_bytes: int = 0
        self.cuda_allocated_bytes: int = 0

    def __str__(self) -> str:
        cuda_res_mb = self.cuda_reserved_bytes / (1024**2)
        cuda_alc_mb = self.cuda_allocated_bytes / (1024**2)
        cost_str = ", ".join([f"{a:.2e}" for a in self.total_cost[-5:]])
        if len(self.total_cost) > 5:
            cost_str = f"... {cost_str}"
        conv_error_str = ", ".join([f"{a:.2e}" for a in self.constraint_violation[-5:]])
        if len(self.constraint_violation) > 5:
            conv_error_str = f"... {conv_error_str}"
        admm_iters_str = ", ".join([f"{a}" for a in self.admm_iters[:]])
        ls_iters_str = ", ".join([f"{a}" for a in self.ls_iters[:]])
        alphas_str = ", ".join([f"{a:.4f}" for a in self.ls_alphas[-5:]])
        if len(self.ls_alphas) > 5:
            alphas_str = f"... {alphas_str}"

        return (
            f"=== SQP Solution Log ===\n"
            f" Envs Terminated        : {self.envs_terminated}\n"
            f" Iterations             : {self.sqp_iterations}\n"
            f" Total Cost             : [{cost_str}]\n"
            f" Conv. Error            : [{conv_error_str}]\n"
            f" Solve Time             : {self.termination_time_s:.4f} s\n"
            f" ADMM Iterations        : [{admm_iters_str}]\n"
            f" Line Search Iterations : [{ls_iters_str}]\n"
            # f" Line Search Alphas : [{alphas_str}]\n"
            f" CUDA Allocated         : {cuda_alc_mb:.2f} MB\n"
            f" CUDA Reserved          :  {cuda_res_mb:.2f} MB\n"
            f"========================="
        )


def sqp_solve(problem: Problem, parameters: SqpParameters, initial_guess: SqpSolution):
    batch_size = problem.batch_size

    terminated = torch.zeros((batch_size), dtype=torch.bool)
    line_search_fails = 0
    current_guess = initial_guess
    best_cost, best_constr_inf = problem.evaluate_guess(current_guess)
    if parameters.ls_function == "merit":
        # Merit function
        merit_mu = parameters.merit_mu
        best_phi = best_cost + parameters.merit_mu * best_constr_inf

    sqp_log = SqpSolutionLog()
    admm_solution = None

    # Solve for sqp_max_iter steps
    t_solve_start = time.time()
    for iter in range(parameters.sqp_max_iter):
        try:
            ## Linearize problem ##
            regularization_scale = line_search_fails * 1e-8
            mat = problem.linearize(current_guess, regularization_scale)

            if (
                admm_solution is None
                and parameters.admm_warm_start
                and parameters.admm_initialize_unconstrained
            ):
                # Solve the unconstrained problem
                admm_solution, admm_log = lqr_solve(problem, mat)

            if parameters.admm_warm_start:
                admm_solution, admm_log = admm_qp_solve(
                    problem, parameters, mat, admm_solution
                )
            else:
                admm_solution, admm_log = admm_qp_solve(problem, parameters, mat)

            # Log admm iterations
            sqp_log.admm_iters.append(admm_log.iterations)

            ## Line search ##
            alpha = torch.ones((batch_size))
            dones = terminated.clone()
            for ls_iter in range(parameters.ls_max_iter):
                new_guess = SqpSolution(
                    x=current_guess.x
                    + torch.einsum("b,bhj->bhj", alpha, admm_solution.dx),
                    u=current_guess.u
                    + torch.einsum("b,bhj->bhj", alpha, admm_solution.du),
                    mu=admm_solution.mu,
                    nu=admm_solution.nu,
                    ksi=admm_solution.ksi,
                )

                # Evaluate current alpha
                cost, constr_inf = problem.evaluate_guess(new_guess)
                # Backtracking line search option
                if parameters.ls_function == "filter":
                    cost_improved = cost < best_cost
                    constr_inf_improved = constr_inf < best_constr_inf
                    update_mask = torch.logical_or(cost_improved, constr_inf_improved)
                # Merit function option
                elif parameters.ls_function == "merit":
                    phi = cost + parameters.merit_mu * constr_inf
                    update_mask = phi < best_phi

                update_mask = update_mask & ~dones
                if update_mask.any():
                    # Update relevant variables
                    current_guess.x[:][update_mask] = new_guess.x[:][update_mask]
                    current_guess.u[:][update_mask] = new_guess.u[:][update_mask]
                    current_guess.mu[:][update_mask] = new_guess.mu[:][update_mask]
                    current_guess.nu[:][update_mask] = new_guess.nu[:][update_mask]
                    # Mark environments as finished
                    dones[update_mask] = True
                    # Update best filter and merit candidates
                    best_cost[update_mask] = cost[update_mask]
                    best_constr_inf[update_mask] = constr_inf[update_mask]
                    if parameters.ls_function == "merit":
                        best_phi[update_mask] = phi[update_mask]

                # Decrease alpha
                alpha[~dones] *= 0.5
                if torch.all(dones):
                    # Reset line search fails
                    line_search_fails = 0
                    break

            sqp_log.ls_iters.append(ls_iter + 1)
            if ls_iter == parameters.ls_max_iter - 1:
                print("Line search failed")
                line_search_fails += 1

            #######################
            ## Check termination ##
            #######################
            """
            Check the KKT conditions:
            - ||L||_inf < eps
            - ||dynamics(x, u) - x_next||_inf < eps
            - ||h(x, u)||_inf < eps
            """

            ## Primal Feasibility ##
            # Computing Lx, Lu is expensive, so we check for stationarity in dx.T @ dx, du.T @ du
            dot_delta_x = torch.einsum(
                "bhi,bhi->bh", admm_solution.dx, admm_solution.dx
            )
            dot_delta_u = torch.einsum(
                "bhi,bhi->bh", admm_solution.du, admm_solution.du
            )
            dx_inf = torch.norm(dot_delta_x, p=float("inf"), dim=[1])
            du_inf = torch.norm(dot_delta_u, p=float("inf"), dim=[1])
            stationarity = torch.logical_and(
                dx_inf < parameters.sqp_eps,
                du_inf < parameters.sqp_eps,
            )

            constraint_satisfaction = best_constr_inf < parameters.sqp_eps

            # terminated = torch.logical_and(stationarity, constraint_satisfaction)
            terminated = constraint_satisfaction
            if terminated.all():
                break
        except KeyboardInterrupt:
            break
    t_solve_end = time.time()

    ##############
    ## Fill log ##
    ##############
    sqp_log.solve_wall_time_s = t_solve_end - t_solve_start
    sqp_log.sqp_iterations = iter + 1
    sqp_log.envs_terminated = torch.count_nonzero(terminated).item()
    sqp_log.total_cost = best_cost.tolist()
    sqp_log.constraint_violation = best_constr_inf
    if torch.get_default_device() != "cpu":
        sqp_log.cuda_reserved_bytes = torch.cuda.memory_reserved(0)
        sqp_log.cuda_allocated_bytes = torch.cuda.memory_allocated(0)
    return current_guess, sqp_log
