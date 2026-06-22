import sys
import time
import torch
from diffsqp.utils.math import mm, mv, inf_norm
from typing import List

from diffsqp.problems import Problem, ProblemParameters
from diffsqp.solvers import Lqr, QP
from dataclasses import dataclass
from diffsqp.types import Trajectory, QpParameters


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
        self.convergence_error: List[float] = []

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
        conv_error_str = ", ".join([f"{a:.2e}" for a in self.convergence_error[-5:]])
        if len(self.convergence_error) > 5:
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


## What to keep as info:
# QP time
# Line search time
# Line search iterations
# Total SQP iterations


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

        # self.admm_solver = Admm(self.prob, qp_solver)
        self.qp_solver = None
        if self.params.qp_solver == "lqr":
            self.qp_solver = Lqr(prob)
        elif self.params.qp_solver == "qp":
            self.qp_solver = QP(prob)

        self.best_cost, self.best_constr_inf = self.calc_metrics_(
            self.current_guess.x,
            self.current_guess.u,
        )
        if self.params.ls_function == "merit":
            # Merit function
            self.merit_mu = self.params.merit_mu
            self.best_phi = self.merit_(self.best_cost, self.best_constr_inf)

        # Book-keeping of correction scales
        self.prev_dx_inf: torch.Tensor = torch.zeros(self.prob.n_batch)
        self.prev_du_inf: torch.Tensor = torch.zeros(self.prob.n_batch)

        self.terminated = torch.zeros((self.prob.n_batch), dtype=torch.bool)

        self.log = SqpSolutionLog()

    def solve(self):
        # Solve for sqp_max_iter steps
        t_solve_start = time.time()
        for iter in range(self.params.sqp_max_iter):
            # Get LQR corrections
            # TODO: Log QP solve time
            # Perform ADMM step
            dx, du, mu_, nu_ = self.qp_solver.solve(self.current_guess)

            # Line search
            # TODO: Log line search time
            ls_info = self.line_search_(
                self.current_guess.x, self.current_guess.u, dx, du, mu_, nu_
            )

            if ls_info["iterations"] == self.params.ls_max_iter - 1:
                print("Line search failed")

            self.log.ls_iters.append(ls_info["iterations"])

            # Check termination
            self.terminated = self.check_termination_(dx, du)
            if self.terminated.all():
                break
        t_solve_end = time.time()

        # Fill log
        self.log.solve_wall_time_s = t_solve_end - t_solve_start
        self.log.sqp_iterations = iter + 1
        self.log.envs_terminated = torch.count_nonzero(self.terminated).item()
        self.log.total_cost = self.best_cost.tolist()
        if torch.get_default_device() != "cpu":
            self.log.cuda_reserved_bytes = torch.cuda.memory_reserved(0)
            self.log.cuda_allocated_bytes = torch.cuda.memory_allocated(0)
        return self.log

    def line_search_(self, x, u, delta_x, delta_u, mu, nu):
        alpha = torch.ones((self.prob.n_batch, 1))
        dones = self.terminated.clone()
        iter = 0
        while (not torch.all(dones)) and (iter < self.params.ls_max_iter):
            iter += 1
            x_, u_ = self.calc_cadidate_solutions_(alpha, x, u, delta_x, delta_u)

            # Evaluate current alpha
            cost, constr_inf = self.calc_metrics_(x_, u_)
            if self.params.ls_function == "filter":
                update_mask = self.evaluate_filter_(cost, constr_inf)
            elif self.params.ls_function == "merit":
                phi = self.merit_(cost, constr_inf)
                update_mask = self.evaluate_merit_(phi)

            update_mask = update_mask & ~dones
            # Update relevant variables
            if update_mask.any():
                self.update_variables_(update_mask, x_, u_, mu, nu)
                # Mark environments as finished
                dones[update_mask] = True

            # Update best filter and merit candidates
            self.best_cost[update_mask] = cost[update_mask]
            self.best_constr_inf[update_mask] = constr_inf[update_mask]
            if self.params.ls_function == "merit":
                self.best_phi[update_mask] = phi[update_mask]

            # Decrease alpha
            alpha[~dones] *= 0.5
        return {"iterations": iter, "alphas": alpha, "dones": dones}

    def evaluate_filter_(self, cost, constr_inf):
        cost_improved = cost < self.best_cost
        constr_inf_improved = constr_inf < self.best_constr_inf
        return cost_improved | constr_inf_improved

    def evaluate_merit_(self, phi):
        return phi < self.best_phi

    def merit_(self, cost, constr_inf):
        # print(cost, constr_inf)
        return cost + self.merit_mu * constr_inf

    def update_variables_(self, update_mask, x_, u_, mu_, nu_):
        for k in range(self.horizon - 1):
            self.current_guess.x[:, k][update_mask] = x_[:, k][update_mask]
            self.current_guess.u[:, k][update_mask] = u_[:, k][update_mask]
            self.current_guess.mu[:, k + 1][update_mask] = mu_[:, k + 1][update_mask]
            if self.prob.underactuation is not None:
                self.current_guess.nu[:, k][update_mask] = nu_[:, k][update_mask]
        self.current_guess.x[:, -1][update_mask] = x_[:, -1][update_mask]

    def normalize_hessians_(dones):
        pass

    def check_termination_(self, delta_x, delta_u):
        """
        Check the KKT conditions:
        - ||L||_inf < eps
        - ||dynamics(x, u) - x_next||_inf < eps
        - ||h(x, u)||_inf < eps

        ! : For the Lagrangian gradient, we only need to include the active constraints
        """

        ## Primal Feasibility ##
        # Computing Lx, Lu is expensive, so we check for stationarity in the QP corrections.

        # Lx, Lu = self.prob.Lx_Lu()
        dot_delta_x = torch.einsum("bhi,bhi->bh", delta_x, delta_x)
        dot_delta_u = torch.einsum("bhi,bhi->bh", delta_u, delta_u)

        ## Constraint Violations ##
        # Dynamics
        dyn = self.dynamics_violation_(
            self.current_guess.x[:, 1:],
            self.current_guess.x[:, :-1],
            self.current_guess.u[:],
        )
        dyn_inf = torch.norm(dyn, p=float("inf"), dim=[1, 2])

        # Underactuation
        if self.prob.underactuation is not None:
            uact = self.underactuation_violation_(
                self.current_guess.x[:, :-1], self.current_guess.u[:]
            )
            uact_inf = torch.norm(uact, p=float("inf"), dim=[1, 2])

        if self.prob.underactuation is not None:
            convergence_error = torch.maximum(dyn_inf, uact_inf)
        else:
            convergence_error = dyn_inf

        # Lx_inf = torch.norm(Lx, p=float("inf"), dim=[0, 2])
        # Lu_inf = torch.norm(Lu, p=float("inf"), dim=[0, 2])
        dx_inf = torch.norm(dot_delta_x, p=float("inf"), dim=[1])
        du_inf = torch.norm(dot_delta_u, p=float("inf"), dim=[1])
        terminate_Lx = (
            torch.norm(torch.sub(dx_inf, self.prev_dx_inf)) < self.params.sqp_eps
        )
        terminate_Lu = (
            torch.norm(torch.sub(du_inf, self.prev_du_inf)) < self.params.sqp_eps
        )
        self.prev_dx_inf = dx_inf
        self.prev_du_inf = du_inf
        terminate_constraints = convergence_error < self.params.sqp_eps

        # Logging
        self.log.convergence_error = convergence_error

        return torch.logical_and(
            terminate_Lx, torch.logical_and(terminate_Lu, terminate_constraints)
        )
        # return terminate_constraints

    def calc_cadidate_solutions_(self, alpha, x, u, delta_x, delta_u):
        x_ = torch.zeros((self.prob.n_batch, self.horizon, self.prob.n_x))
        u_ = torch.zeros((self.prob.n_batch, self.horizon - 1, self.prob.n_u))
        # TODO: This is vectorizable
        for k in range(self.horizon):
            x_[:, k] = x[:, k] + torch.mul(alpha, delta_x[:, k])
            if k < self.horizon - 1:
                u_[:, k] = u[:, k] + torch.mul(alpha, delta_u[:, k])
        return x_, u_

    def calc_metrics_(self, x_cand, u_cand):
        # Returns:
        # cost: total cost
        # constr_inf: ||constr||_inf
        cost = self.total_cost_(x_cand, u_cand)
        dyn = self.dynamics_violation_(x_cand[:, 1:], x_cand[:, :-1], u_cand[:])
        dyn_inf = torch.norm(dyn, p=float("inf"), dim=[1, 2])

        if self.prob.underactuation is None:
            return cost, dyn_inf
        else:
            uact = self.underactuation_violation_(x_cand[:, :-1], u_cand[:])
            uact_inf = torch.norm(uact, p=float("inf"), dim=[1, 2])
            return cost, torch.maximum(dyn_inf, uact_inf)

    def total_cost_(self, x, u):
        cost = torch.zeros((self.prob.n_batch))
        for k in range(self.horizon - 1):
            # Calculate total trajectory cost
            cost += self.prob.l(k, x[:, k], u[:, k])
        # Add final node cost
        cost += self.prob.l(-1, x[:, -1])

        return cost

    def dynamics_violation_(self, x_next, x, u):
        f = self.prob.dynamics.f
        return x_next - f(x, u, self.prob.dt)

    def underactuation_violation_(self, x, u):
        h = self.prob.underactuation.h
        return h(x, u)
