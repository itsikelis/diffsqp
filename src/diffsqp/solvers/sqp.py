import sys
import json
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
        self.admm_alpha: float = args["admm_alpha"]
        self.admm_sigma: float = args["admm_sigma"]
        # Rho related #
        self.admm_update_rho: bool = args["admm_update_rho"]
        self.admm_reset_rho: float = args["admm_reset_rho"]
        self.admm_rho_init: float = args["admm_rho_init"]
        self.admm_rho_min: torch.Tensor = torch.tensor([args["admm_rho_min"]])
        self.admm_rho_max: torch.Tensor = torch.tensor([args["admm_rho_max"]])
        self.admm_adaptive_rho_tolerance = args["admm_adaptive_rho_tolerance"]
        self.admm_rho_update_iter_freq = args["admm_rho_update_iter_freq"]
        # Warm-start related #
        self.admm_warm_start_unconstrained: float = args[
            "admm_warm_start_unconstrained"
        ]
        self.admm_reset_ksi: float = args["admm_reset_ksi"]
        self.admm_abs_tolerance = args["admm_abs_tolerance"]
        self.admm_abs_tolerance_final = args["admm_abs_tolerance_final"]
        self.admm_rel_tolerance = args["admm_rel_tolerance"]
        self.admm_rel_tolerance_final = args["admm_rel_tolerance_final"]
        self.admm_tolerance_update_steps = args["admm_tolerance_update_steps"]

        self.sqp_max_iter: int = args["sqp_max_iter"]
        self.lqr_reg_init: int = args["lqr_reg_init"]
        self.armijo_beta: float = args["armijo_beta"]
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

        self.admm_iter_hist: List[float] = []
        self.ls_iter_hist: List[float] = []
        self.cost_hist: List[List[float]] = []
        self.dynamics_violation_hist: List[List[float]] = []
        self.constraint_violation_hist: List[float] = []

        # GPU related
        self.cuda_reserved_bytes: int = 0
        self.cuda_allocated_bytes: int = 0

    def __str__(self) -> str:
        cuda_res_mb = self.cuda_reserved_bytes / (1024**2)
        cuda_alc_mb = self.cuda_allocated_bytes / (1024**2)

        # Helper to safely extract medians from batched tensors
        def get_medians(hist):
            return [torch.median(x).item() for x in hist]

        # Helper to format lists (displays up to the last 5 elements)
        def fmt_list(lst, fmt="{:.2e}"):
            s = ", ".join([fmt.format(a) for a in lst[-5:]])
            return f"... {s}" if len(lst) > 5 else s

        # Apply formatting
        cost_str = fmt_list(self.total_cost)
        conv_error_str = fmt_list(self.constraint_violation)

        # Note: Fixed the attribute names to match your __init__ (admm_iter_hist instead of admm_iters)
        admm_iters_str = fmt_list(self.admm_iter_hist, fmt="{}")
        ls_iters_str = fmt_list(self.ls_iter_hist, fmt="{}")

        return (
            f"=== SQP Solution Log ===\n"
            f" Envs Terminated        : {self.envs_terminated}\n"
            f" Iterations             : {self.sqp_iterations}\n"
            f" Total Cost             : [{cost_str}]\n"
            f" Conv. Error            : [{conv_error_str}]\n"
            f" Solve Time             : {self.solve_wall_time_s:.4f} s\n"
            f" ADMM Iterations        : [{admm_iters_str}]\n"
            f" Line Search Iterations : [{ls_iters_str}]\n"
            f" CUDA Allocated         : {cuda_alc_mb:.2f} MB\n"
            f" CUDA Reserved          :  {cuda_res_mb:.2f} MB\n"
            f"========================="
        )

    def save_to_json(self, filepath: str) -> None:
        """Saves the current state of the log to a JSON file."""
        with open(filepath + ".json", "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=4)


def sqp_solve(problem: Problem, parameters: SqpParameters, initial_guess: SqpSolution):
    batch_size = problem.batch_size

    terminated = torch.zeros((batch_size), dtype=torch.bool)
    reg_incr = torch.zeros((batch_size,))
    current_guess = initial_guess
    # Cost, Dynamics Violation, Constraint Violation, Complementarity Slackness
    best_cost, best_dyn_inf, best_constr_inf, best_comp_inf = problem.evaluate_guess(
        current_guess
    )
    if parameters.ls_function == "merit":
        # Merit function
        merit_mu = parameters.merit_mu
        convergence_error = torch.maximum(best_dyn_inf, best_constr_inf)
        best_phi = best_cost + parameters.merit_mu * convergence_error

    sqp_log = SqpSolutionLog()
    admm_solution = None
    admm_log = None

    # Solve for sqp_max_iter steps
    t_solve_start = time.time()
    for sqp_iter in range(parameters.sqp_max_iter):
        try:
            # Base LQR damping scaled per batch element: 1e-5 * 10^(reg_incr)
            lqr_reg = parameters.lqr_reg_init * torch.pow(10.0, reg_incr)

            ## Linearize problem ##
            mat = problem.linearize(current_guess)

            admm_solution, admm_log = admm_qp_solve(
                problem, parameters, mat, lqr_reg, admm_solution
            )

            #################
            ## Line search ##
            #################

            # Directional Derivative for Armijo check
            if parameters.ls_function == "merit":
                l_dir_deriv = problem.evaluate_directional_derivatives(
                    current_guess, admm_solution
                )

            alpha = torch.ones((batch_size))
            dones = terminated.detach().clone()
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
                cost, dyn_inf, constr_inf, comp_inf = problem.evaluate_guess(new_guess)
                # Backtracking line search option
                if parameters.ls_function == "filter":
                    cost_improved = cost < best_cost
                    dyn_inf_improved = dyn_inf < best_dyn_inf
                    constr_inf_improved = constr_inf < best_constr_inf
                    # print(
                    #     f"Cost:, {cost.tolist()} \ {best_cost.tolist()} Dyn inf: {dyn_inf.tolist()} \ {best_dyn_inf.tolist()} Dyn inf: {constr_inf.tolist()} \ {best_constr_inf.tolist()}"
                    # )
                    update_mask = cost_improved | dyn_inf_improved | constr_inf_improved
                # Merit function option
                elif parameters.ls_function == "merit":
                    convergence_error = torch.maximum(dyn_inf, constr_inf)
                    phi = cost + parameters.merit_mu * convergence_error
                    armijo_threshold = (
                        best_phi + parameters.armijo_beta * alpha * l_dir_deriv
                    )
                    update_mask = phi < armijo_threshold

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
                    best_dyn_inf[update_mask] = dyn_inf[update_mask]
                    best_constr_inf[update_mask] = constr_inf[update_mask]
                    best_comp_inf[update_mask] = comp_inf[update_mask]
                    if parameters.ls_function == "merit":
                        best_phi[update_mask] = phi[update_mask]
                    # print(
                    #     "SQP Iter: ",
                    #     sqp_iter,
                    #     "LS Iter: ",
                    #     ls_iter,
                    #     "Cost: ",
                    #     best_cost,
                    #     "Conv Error: ",
                    #     best_dyn_inf,
                    #     best_constr_inf,
                    # )

                # Decrease alpha
                alpha[~dones] *= 0.5
                if torch.all(dones):
                    break

            ls_failed = (~dones) & (~terminated)
            if ls_failed.any():
                print("Line search failed")
                reg_incr[ls_failed] += 1.0
                reg_incr[~ls_failed] = 0.0
            else:
                reg_incr.zero_()

            # Iteration log #
            sqp_log.admm_iter_hist.append(admm_log.iterations)
            sqp_log.ls_iter_hist.append(ls_iter + 1)
            sqp_log.cost_hist.append(best_cost.tolist())
            sqp_log.dynamics_violation_hist.append(best_dyn_inf.tolist())
            sqp_log.constraint_violation_hist.append(best_constr_inf.tolist())

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

            convergence_error = torch.maximum(best_dyn_inf, best_constr_inf)
            constraint_satisfaction = convergence_error < parameters.sqp_eps

            complementarity = best_comp_inf < parameters.sqp_eps

            terminated = stationarity & constraint_satisfaction & complementarity
            # terminated = constraint_satisfaction & complementarity

            if terminated.all():
                break
        except KeyboardInterrupt:
            break
    t_solve_end = time.time()

    ##############
    ## Fill log ##
    ##############
    sqp_log.solve_wall_time_s = t_solve_end - t_solve_start
    sqp_log.sqp_iterations = sqp_iter + 1
    sqp_log.envs_terminated = torch.count_nonzero(terminated).item()
    sqp_log.total_cost = best_cost.tolist()
    sqp_log.constraint_violation = best_constr_inf.tolist()
    if torch.get_default_device() != "cpu":
        sqp_log.cuda_reserved_bytes = torch.cuda.memory_reserved(0)
        sqp_log.cuda_allocated_bytes = torch.cuda.memory_allocated(0)
    return current_guess, sqp_log
