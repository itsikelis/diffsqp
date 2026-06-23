import sys
import time
import torch
from diffsqp.utils.math import mm, mv, inf_norm
from typing import List

from diffsqp.problems import Problem, ProblemParameters
from diffsqp.solvers import QP
from diffsqp.solvers import lqr_forward_pass, lqr_backward_pass
from dataclasses import dataclass
from diffsqp.types import Trajectory, QpParameters, QpSolution


class SqpParameters:
    def __init__(self, **args):
        self.sqp_max_iter: int = args["sqp_max_iter"]
        self.merit_mu: float = args["merit_mu"]
        self.ls_max_iter: int = args["ls_max_iter"]
        self.sqp_eps: float = args["sqp_eps"]
        self.qp_solver: str = args["qp_solver"]
        self.ls_function: str = args["ls_function"]

    def __str__(self) -> str:
        return (
            f"=== SQP Parameters ===\n"
            f"  QP Solver       : {self.qp_solver}\n"
            f"  Line Search Fn  : {self.ls_function}\n"
            f"  SQP Max Iter    : {self.sqp_max_iter}\n"
            f"  Line Search Max : {self.ls_max_iter}\n"
            f"  SQP Tolerance   : {self.sqp_eps:.2e}\n"
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
        iters_str = ", ".join([f"{a}" for a in self.ls_iters[:]])
        alphas_str = ", ".join([f"{a:.4f}" for a in self.ls_alphas[-5:]])
        if len(self.ls_alphas) > 5:
            alphas_str = f"... {alphas_str}"

        return (
            f"=== SQP Solution Log ===\n"
            f"  Envs Terminated    : {self.envs_terminated}\n"
            f"  Iterations         : {self.sqp_iterations}\n"
            f"  Total Cost         : [{cost_str}]\n"
            f"  Conv. Error        : [{conv_error_str}]\n"
            f"  Solve Time         : {self.termination_time_s:.4f} s\n"
            f"  Line Search Iters  : [{iters_str}]\n"
            # f"  Line Search Alphas : [{alphas_str}]\n"
            f"  CUDA Allocated     : {cuda_alc_mb:.2f} MB\n"
            f"  CUDA Reserved      :  {cuda_res_mb:.2f} MB\n"
            f"========================="
        )


# Return total trajectory cost and constraint violations
def evaluate_trajectory(problem: Problem, trajectory: Trajectory):
    batch_size = problem.n_batch
    horizon = problem.horizon
    dt = problem.dt

    # Calculate total trajectory cost
    cost = torch.zeros((batch_size))
    for k in range(horizon - 1):
        cost += problem.l(k, trajectory.x[:, k], trajectory.u[:, k])
    cost += problem.l(-1, trajectory.x[:, -1])

    # Dynamics violation
    x_next = trajectory.x[:, 1:]
    x_curr = trajectory.x[:, :-1]
    u_curr = trajectory.u[:]
    dynamics_violations = x_next - problem.dynamics.f(x_curr, u_curr, dt)
    max_dynamics_violation = torch.norm(dynamics_violations, p=float("inf"), dim=[1, 2])

    # Underactuation violation
    if problem.underactuation is not None:
        uact_violation = problem.underactuation.h(x_curr, u_curr)
        max_uact_violation = torch.norm(uact_violation, p=float("inf"), dim=[1, 2])

    if problem.underactuation is None:
        return cost, max_dynamics_violation
    else:
        return cost, torch.maximum(max_dynamics_violation, max_uact_violation)


## What to keep as info:
# QP time
# Line search time
# Line search iterations
# Total SQP iterations


def get_linearized_matrices(prob: Problem, trajectory: Trajectory):
    batch_size = prob.n_batch
    horizon = prob.horizon
    n_x, n_u = prob.n_x, prob.n_u
    n_h = prob.n_h

    Q = torch.zeros((batch_size, horizon, n_x, n_x))
    q = torch.zeros((batch_size, horizon, n_x))
    R = torch.zeros((batch_size, horizon - 1, n_u, n_u))
    r = torch.zeros((batch_size, horizon - 1, n_u))
    S = torch.zeros((batch_size, horizon - 1, n_u, n_x))

    A = torch.zeros((batch_size, horizon - 1, n_x, n_x))
    B = torch.zeros((batch_size, horizon - 1, n_x, n_u))
    b = torch.zeros((batch_size, horizon - 1, n_x))

    C = None
    D = None
    d = None
    if prob.underactuation is not None:
        n_h = prob.n_h
        C = torch.zeros((batch_size, horizon - 1, n_h, n_x))
        D = torch.zeros((batch_size, horizon - 1, n_h, n_u))
        d = torch.zeros((batch_size, horizon - 1, n_h))

    # Fill matrices
    for i in range(horizon - 1):
        x_lin, u_lin, x_next = (
            trajectory.x[:, i],
            trajectory.u[:, i],
            trajectory.x[:, i + 1],
        )

        A[:, i] = prob.dynamics.fx(x_lin, u_lin, prob.dt)
        B[:, i] = prob.dynamics.fu(x_lin, u_lin, prob.dt)
        b[:, i] = prob.dynamics.f(x_lin, u_lin, prob.dt) - x_next

        Q[:, i] = prob.lxx(i, x_lin, u_lin)
        q[:, i] = prob.lx(i, x_lin, u_lin)
        R[:, i] = prob.luu(i, x_lin, u_lin)
        r[:, i] = prob.lu(i, x_lin, u_lin)
        S[:, i] = prob.lux(i, x_lin, u_lin)

        # Underactuation augmentation
        if prob.underactuation is not None:
            C[:, i] = prob.underactuation.hx(x_lin, u_lin)
            D[:, i] = prob.underactuation.hu(x_lin, u_lin)
            d[:, i] = prob.underactuation.h(x_lin, u_lin)

    x_F = trajectory.x[:, -1]
    Q[:, -1] = prob.lxx(-1, x_F)
    q[:, -1] = prob.lx(-1, x_F)

    return QpParameters(Q=Q, q=q, R=R, r=r, S=S, A=A, B=B, b=b, C=C, D=D, d=d)


def sqp_solve(problem: Problem, parameters: SqpParameters, initial_guess: Trajectory):
    batch_size = problem.n_batch

    terminated = torch.zeros((batch_size), dtype=torch.bool)
    current_guess = initial_guess
    best_cost, best_constr_inf = evaluate_trajectory(problem, current_guess)
    if parameters.ls_function == "merit":
        # Merit function
        merit_mu = parameters.merit_mu
        best_phi = best_cost + parameters.merit_mu * best_constr_inf

    log = SqpSolutionLog()
    # Solve for sqp_max_iter steps
    t_solve_start = time.time()
    for iter in range(parameters.sqp_max_iter):
        # Linearize problem
        mat = get_linearized_matrices(problem, current_guess)
        # Get LQR corrections
        # dx, du, mu_, nu_ = self.qp_solver.solve(self.current_guess, mat)
        K, k, P, p = lqr_backward_pass(problem, mat)
        corrections = lqr_forward_pass(problem, K, k, P, p, mat.A, mat.B, mat.b)

        # Line search
        # TODO: Log line search time
        # ls_info = self.line_search_(problem, parameters, current_guess, corrections)
        alpha = torch.ones((batch_size))
        dones = terminated.clone()
        for ls_iter in range(parameters.ls_max_iter):
            new_guess = Trajectory(
                x=current_guess.x + torch.einsum("b,bhj->bhj", alpha, corrections.dx),
                u=current_guess.u + torch.einsum("b,bhj->bhj", alpha, corrections.du),
                mu=corrections.mu,
                nu=corrections.nu,
                lam=None,
            )

            # Evaluate current alpha
            cost, constr_inf = evaluate_trajectory(problem, new_guess)
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
                break

        log.ls_iters.append(ls_iter + 1)
        if ls_iter == parameters.ls_max_iter - 1:
            print("Line search failed")

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
        dot_delta_x = torch.einsum("bhi,bhi->bh", corrections.dx, corrections.dx)
        dot_delta_u = torch.einsum("bhi,bhi->bh", corrections.du, corrections.du)
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
    t_solve_end = time.time()

    ##############
    ## Fill log ##
    ##############
    log.solve_wall_time_s = t_solve_end - t_solve_start
    log.sqp_iterations = iter + 1
    log.envs_terminated = torch.count_nonzero(terminated).item()
    log.total_cost = best_cost.tolist()
    log.constraint_violation = best_constr_inf
    if torch.get_default_device() != "cpu":
        log.cuda_reserved_bytes = torch.cuda.memory_reserved(0)
        log.cuda_allocated_bytes = torch.cuda.memory_allocated(0)
    return current_guess, log


class Sqp:
    def __init__(
        self,
        prob: Problem,
        params: SqpParameters,
        init_guess: Trajectory,
    ) -> None:
        self.prob = prob
        self.params = params
        self.horizon = self.prob.horizon

        self.current_guess = init_guess

        if self.params.ls_function == "merit":
            # Merit function
            self.merit_mu = self.params.merit_mu
            self.best_phi = self.merit_(self.best_cost, self.best_constr_inf)

        self.terminated = torch.zeros((self.prob.n_batch), dtype=torch.bool)

    def solve(self):
        return self.solve_(self.prob, self.params, self.current_guess)
