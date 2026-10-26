import sys
import json
import time
import torch
from copy import copy
from diffsqp.utils.math import mm, mv, inf_norm
from typing import List

from diffsqp.problems import Problem, ProblemParameters
from diffsqp.solvers import QP
from diffsqp.solvers import admm_qp_solve, admm_qp_resume, lqr_solve
from diffsqp.solvers.admm_qp import constraint_map
from diffsqp.solvers.admm_bsqp import (
    BsqpState,
    bsqp_equality_correction,
    bsqp_qp_resume,
    bsqp_qp_solve,
)
from diffsqp.solvers.lqr import lqr_with_offsets, lqr_solve_factorized
from diffsqp.solvers.line_search import make_line_search
from diffsqp.utils.math import bmv
from dataclasses import dataclass
from diffsqp.types import SqpSolution, AdmmSolution, LqrSolution


class SqpParameters:
    def __init__(self, **args):
        # ADMM implementation: "bsqp" (C++/CUDA kernels) or "torch"
        self.admm_backend: str = args.get("admm_backend", "bsqp")
        # bsqp only: residual / termination check period, in ADMM iterations
        self.admm_check_every: int = args.get("admm_check_every", 1)
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
        # Start the consensus variables z at the projection of the warm start
        self.admm_warm_start_z: bool = args.get("admm_warm_start_z", False)
        # Inexact QP safeguard: resume ADMM (up to admm_max_resumes times) for
        # elements whose step does not reduce the linearized constraint
        # violation to admm_resume_kappa * current violation. 0 disables it.
        self.admm_resume_kappa: float = args.get("admm_resume_kappa", 0.0)
        self.admm_max_resumes: int = args.get("admm_max_resumes", 3)
        self.admm_abs_tolerance = args["admm_abs_tolerance"]
        self.admm_abs_tolerance_final = args["admm_abs_tolerance_final"]
        self.admm_rel_tolerance = args["admm_rel_tolerance"]
        self.admm_rel_tolerance_final = args["admm_rel_tolerance_final"]
        self.admm_tolerance_update_steps = args["admm_tolerance_update_steps"]

        self.sqp_max_iter: int = args["sqp_max_iter"]
        self.lqr_reg_init: int = args["lqr_reg_init"]
        # Solve the LQR recursions with a parallel scan (fewer, larger ops)
        self.lqr_parallel_scan: bool = args.get("lqr_parallel_scan", False)
        self.armijo_beta: float = args["armijo_beta"]
        self.merit_mu: float = args["merit_mu"]
        # Initial penalty of the "merit_adaptive" line search (it only grows)
        self.merit_mu_init: float = args.get("merit_mu_init", 1.0)
        self.ls_max_iter: int = args["ls_max_iter"]
        self.check_complementarity: float = args["check_complementarity"]
        # Multiply the LQR regularization by 10 after a failed line search
        self.ls_fail_increase_reg: bool = args.get("ls_fail_increase_reg", True)
        # Try a second-order correction when the full step is rejected
        self.second_order_correction: bool = args.get("second_order_correction", False)
        self.sqp_cost_eps: float = args["sqp_cost_eps"]
        self.sqp_viol_eps: float = args["sqp_viol_eps"]
        self.qp_solver: str = args["qp_solver"]
        self.ls_function: str = args["ls_function"]
        # Print progress (line search acceptances, failures, rho updates)
        self.verbose: bool = args.get("verbose", True)

    def __str__(self) -> str:
        return (
            f"=== SQP Parameters ===\n"
            f" QP Solver       : {self.qp_solver}\n"
            f" Line Search Fn  : {self.ls_function}\n"
            f" SQP Max Iter    : {self.sqp_max_iter}\n"
            f" Line Search Max : {self.ls_max_iter}\n"
            f" SQP Cost Tol.   : {self.sqp_cost_eps:.2e}\n"
            f" SQP Viol. Tol.  : {self.sqp_viol_eps:.2e}\n"
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


def _step(guess: SqpSolution, alpha: torch.Tensor, direction: AdmmSolution):
    """Trial point guess + alpha * direction, with the QP multipliers."""
    a = alpha.view(-1, 1, 1)
    return SqpSolution(
        x=guess.x + a * direction.dx,
        u=guess.u + a * direction.du,
        mu=direction.mu,
        nu=direction.nu,
        ksi=direction.ksi,
    )


def _select(mask: torch.Tensor, new: SqpSolution, old: SqpSolution):
    """Per batch element choice between two guesses (new where mask)."""
    m = mask.view(-1, 1, 1)
    ksi = new.ksi
    if torch.is_tensor(old.ksi):
        ksi = torch.where(m, new.ksi, old.ksi)
    return SqpSolution(
        x=torch.where(m, new.x, old.x),
        u=torch.where(m, new.u, old.u),
        mu=torch.where(m, new.mu, old.mu),
        nu=torch.where(m, new.nu, old.nu),
        ksi=ksi,
    )


def _step_curvature(mat, direction: AdmmSolution):
    """d^T H d of the QP cost Hessian along the step."""
    dx, du = direction.dx, direction.du
    dx_head = dx[:, :-1]
    curvature = torch.sum(dx * bmv(mat.Q, dx), dim=(1, 2))
    curvature += torch.sum(du * bmv(mat.R, du), dim=(1, 2))
    curvature += 2.0 * torch.sum(du * bmv(mat.S, dx_head), dim=(1, 2))
    return curvature


def _linearized_violation(mat, direction: AdmmSolution):
    """Stage constraint violation of the linearized constraints after the
    full step. Dynamics and underactuation are enforced exactly by the
    Riccati recursion, so only the ADMM-handled bounds can remain violated."""
    if mat.lb.shape[-1] == 0:
        return torch.zeros(direction.dx.shape[0])
    g = mat.n + constraint_map(mat, direction.dx, direction.du)
    viol = torch.maximum(torch.relu(mat.lb - g), torch.relu(g - mat.ub))
    return viol.amax(dim=(1, 2))


def _second_order_correction(problem, direction, admm_log, trial):
    """Second-order correction of a rejected full step (Nocedal & Wright 18.3).

    The equality constraints (dynamics and underactuation) are re-linearized
    residuals at the trial point: the correction d_c solves A d_c = -c(x + d)
    with the minimum (ADMM-augmented) cost. The QP matrices are unchanged, so
    the last ADMM factorization is reused: one cheap LQR solve.
    """
    gaps, h = problem.equality_residuals(trial.x, trial.u)
    if isinstance(admm_log.factorization, BsqpState):
        dx_c, du_c = bsqp_equality_correction(admm_log.factorization, gaps, h)
    else:
        fact = lqr_with_offsets(admm_log.factorization, gaps, h)
        correction = lqr_solve_factorized(
            fact, torch.zeros_like(direction.dx), torch.zeros_like(direction.du)
        )
        dx_c, du_c = correction.dx, correction.du
    return AdmmSolution(
        dx=direction.dx + dx_c,
        du=direction.du + du_c,
        mu=direction.mu,
        nu=direction.nu,
        z=direction.z,
        ksi=direction.ksi,
        rho=direction.rho,
        rho_inv=direction.rho_inv,
        rho_common=direction.rho_common,
    )


def sqp_solve(problem: Problem, parameters: SqpParameters, initial_guess: SqpSolution):
    batch_size = problem.batch_size
    verbose = parameters.verbose
    if parameters.admm_backend == "bsqp":
        qp_solve, qp_resume = bsqp_qp_solve, bsqp_qp_resume
    elif parameters.admm_backend == "torch":
        qp_solve, qp_resume = admm_qp_solve, admm_qp_resume
    else:
        raise ValueError(f"unknown admm_backend {parameters.admm_backend!r}")

    # Constraints may have changed since the last solve
    problem.padded_bounds(refresh=True)

    terminated = torch.zeros((batch_size), dtype=torch.bool)
    reg_incr = torch.zeros((batch_size,))
    current_guess = initial_guess
    # Cost, Dynamics Violation, Constraint Violation, Complementarity Slackness
    best_cost, best_dyn_inf, best_constr_inf, best_comp_inf = problem.evaluate_guess(
        current_guess
    )
    line_search = make_line_search(
        parameters, batch_size, best_cost, best_dyn_inf, best_constr_inf
    )

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

            admm_solution, admm_log = qp_solve(
                problem, parameters, mat, lqr_reg, admm_solution
            )

            # Inexact QP safeguard
            theta = torch.maximum(best_dyn_inf, best_constr_inf)
            theta_lin = _linearized_violation(mat, admm_solution)
            if parameters.admm_resume_kappa > 0.0:
                for _ in range(parameters.admm_max_resumes):
                    inaccurate = (
                        (theta_lin > parameters.admm_resume_kappa * theta)
                        & (theta > parameters.sqp_viol_eps)
                        & ~terminated
                        & ~admm_log.failed
                    )
                    if not inaccurate.any():
                        break
                    admm_solution, admm_log = qp_resume(
                        problem,
                        parameters,
                        mat,
                        lqr_reg,
                        admm_solution,
                        admm_log,
                        inaccurate,
                        parameters.admm_max_iter,
                    )
                    theta_lin = _linearized_violation(mat, admm_solution)

            #################
            ## Line search ##
            #################
            alpha = torch.ones((batch_size))

            # If not line search, accept change and break
            if parameters.ls_max_iter == 0:
                current_guess = _step(current_guess, alpha, admm_solution)
                break

            # Directional derivative of the cost and curvature along the step
            grad_dot_d = problem.evaluate_directional_derivatives(
                current_guess, admm_solution, mat
            )
            dHd = _step_curvature(mat, admm_solution)
            line_search.begin(
                sqp_iter,
                best_cost,
                best_dyn_inf,
                best_constr_inf,
                grad_dot_d,
                dHd,
                theta_lin,
            )

            # Elements whose QP failed take no step
            dones = terminated | admm_log.failed
            for ls_iter in range(parameters.ls_max_iter):
                new_guess = _step(current_guess, alpha, admm_solution)

                # Evaluate current alpha
                cost, dyn_inf, constr_inf, comp_inf = problem.evaluate_guess(new_guess)
                update_mask = line_search.accept(
                    alpha, cost, dyn_inf, constr_inf, active=~dones
                )

                # Full step rejected: try the second-order corrected step
                rejected = ~update_mask & ~dones
                if (
                    ls_iter == 0
                    and parameters.second_order_correction
                    and rejected.any()
                ):
                    soc_direction = _second_order_correction(
                        problem, admm_solution, admm_log, new_guess
                    )
                    soc_guess = _step(current_guess, alpha, soc_direction)
                    soc_eval = problem.evaluate_guess(soc_guess)
                    soc_mask = line_search.accept(alpha, *soc_eval[:3], active=rejected)
                    new_guess = _select(soc_mask, soc_guess, new_guess)
                    cost, dyn_inf, constr_inf, comp_inf = (
                        torch.where(soc_mask, a, b)
                        for a, b in zip(soc_eval, (cost, dyn_inf, constr_inf, comp_inf))
                    )
                    update_mask = update_mask | soc_mask

                # Update relevant variables
                current_guess = _select(update_mask, new_guess, current_guess)
                # Mark environments as finished
                dones = dones | update_mask
                # Update best filter and merit candidates
                best_cost = torch.where(update_mask, cost, best_cost)
                best_dyn_inf = torch.where(update_mask, dyn_inf, best_dyn_inf)
                best_constr_inf = torch.where(update_mask, constr_inf, best_constr_inf)
                best_comp_inf = torch.where(update_mask, comp_inf, best_comp_inf)

                # Decrease alpha
                alpha = torch.where(dones, alpha, 0.5 * alpha)
                if torch.all(dones):
                    break
            line_search.end()

            if verbose:
                print(
                    "SQP Iter: ",
                    sqp_iter,
                    "LS Iter: ",
                    ls_iter,
                    "Cost: ",
                    torch.max(best_cost),
                    "Conv Error: ",
                    torch.max(best_dyn_inf),
                    torch.max(best_constr_inf),
                )

            ls_failed = (~dones | admm_log.failed) & (~terminated)
            if not parameters.ls_fail_increase_reg:
                # Only failed factorizations need more regularization
                ls_failed = admm_log.failed & ~terminated
            if ls_failed.any():
                if verbose:
                    print("Line search failed")
                reg_incr = torch.where(ls_failed, reg_incr + 1.0, 0.0)
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
            dx_inf = torch.sum(admm_solution.dx**2, dim=-1).amax(dim=1)
            du_inf = torch.sum(admm_solution.du**2, dim=-1).amax(dim=1)
            stationarity = torch.logical_and(
                dx_inf < parameters.sqp_cost_eps,
                du_inf < parameters.sqp_cost_eps,
            )

            convergence_error = torch.maximum(best_dyn_inf, best_constr_inf)
            constraint_satisfaction = convergence_error < parameters.sqp_viol_eps

            terminated = stationarity & constraint_satisfaction
            if parameters.check_complementarity:
                complementarity = best_comp_inf < parameters.sqp_viol_eps
                terminated &= complementarity

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
