import sys
import time
import torch
from diffsqp.utils.math import mm, mv, inf_norm
from typing import List

from diffsqp.problems import Problem, ProblemParams
from diffsqp.solvers import Lqr, QP
from dataclasses import dataclass


class SqpParams:
    def __init__(self, **args):
        self.sqp_max_iter: int = args["sqp_max_iter"]
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

        self.total_cost: float = 0.0
        self.convergence_error: float = 0.0

        self.solve_wall_time_s: int = 0
        self.sqp_iterations: int = 0

        self.termination_time_s: float = 0.0
        self.ls_alphas: List[float] = []

        # GPU related
        self.cuda_reserved_bytes: int = 0
        self.cuda_allocated_bytes: int = 0

    def __str__(self) -> str:
        cuda_res_mb = self.cuda_reserved_bytes / (1024**2)
        cuda_alc_mb = self.cuda_allocated_bytes / (1024**2)
        alphas_str = ", ".join([f"{a:.4f}" for a in self.ls_alphas[-5:]])
        if len(self.ls_alphas) > 5:
            alphas_str = f"... {alphas_str}"

        return (
            f"=== SQP Solution Log ===\n"
            f"  Envs Terminated : {self.envs_terminated}\n"
            f"  Iterations      : {self.sqp_iterations}\n"
            f"  Total Cost      : {self.total_cost:.6f}\n"
            f"  Conv. Error     : {self.convergence_error:.6e}\n"
            f"  Solve Time      : {self.termination_time_s:.4f} s\n"
            f"  Recent Alphas   : [{alphas_str}]\n"
            f"  CUDA Allocated  : {cuda_alc_mb:.2f} MB\n"
            f"  CUDA Reserved   : {cuda_res_mb:.2f} MB\n"
            f"========================="
        )


## What to keep as info:
# QP time
# Line search time
# Line search iterations
# Total SQP iterations


class Sqp:
    def __init__(self, prob: Problem, params: SqpParams) -> None:
        self.prob = prob
        self.params = params
        self.horizon = self.prob.horizon

        # self.admm_solver = Admm(self.prob, qp_solver)
        self.qp_solver = None
        if self.params.qp_solver == "lqr":
            self.qp_solver = Lqr(prob)
        elif self.params.qp_solver == "qp":
            self.qp_solver = QP(prob)

        if self.params.ls_function == "filter":
            # Log best line search metrics
            self.best_cost, self.best_gamma, self.best_uact = self.calc_metrics(
                self.prob.states,
                self.prob.controls,
            )
        elif self.params.ls_function == "merit":
            # Merit function
            self.ls_mu = 1.0
            self.best_phi = self.merit(self.prob.states, self.prob.controls)

        self.terminated = torch.zeros((self.prob.n_batch), dtype=torch.bool)

        self.log = SqpSolutionLog()

    def solve(self):
        # Solve for sqp_max_iter steps
        t_solve_start = time.time()
        for iter in range(self.params.sqp_max_iter):
            # Get LQR corrections
            # Perform ADMM step
            # TODO: Log QP solve time
            delta_x_qp, delta_u_qp, pi_qp, ni_qp = self.qp_solver.solve()

            # Line search
            # TODO: Log line search time and total iters
            ls_iters, done = self.line_search(delta_x_qp, delta_u_qp, pi_qp, ni_qp)

            # Check termination
            if self.check_termination(delta_x_qp, delta_u_qp):
                break
        t_solve_end = time.time()

        # Fill log
        self.log.solve_wall_time_s = t_solve_end - t_solve_start
        self.log.sqp_iterations = iter + 1
        self.log.envs_terminated = torch.count_nonzero(self.terminated).item()
        if torch.get_default_device() != "cpu":
            self.log.cuda_reserved_bytes = torch.cuda.memory_reserved(0)
            self.log.cuda_allocated_bytes = torch.cuda.memory_allocated(0)
        return self.log

    def merit(self, x, u):
        cost, dyn_inf, uact_inf = self.calc_metrics(x, u)
        # idx = uact > dyn
        # dyn[idx] = uact[idx]
        return cost + self.ls_mu * dyn_inf + self.ls_mu * uact_inf

    def line_search_(self, x, u, delta_x, delta_u, mu, nu):
        alpha = torch.ones((self.prob.n_batch, 1))
        dones = self.terminated.clone()
        iter = 0
        while (not torch.all(dones)) and (iter < self.params.ls_max_iter):
            iter += 1
            x_, u_ = self.calc_cadidate_solutions(alpha, x, u, delta_x, delta_u)

            if self.params.ls_function == "filter":
                update_mask = self.evaluate_filter_(x_, u_, x, u)
            elif self.params.ls_function == "merit":
                update_mask = self.evaluate_merit_(x_, u_, x, u)

            # Update relevant variables
            if update_mask.any():
                self.update_variables_(update_mask, x_, u_, mu, nu)
                dones[update_mask] = True

            # Decrease alpha
            alpha[~update_mask] *= 0.5
        return dones, iter

    def normalize_hessians_(dones):
        pass

    def line_search(self, delta_x, delta_u, pi, ni):
        alpha = torch.ones((self.prob.n_batch, 1))
        dones = self.terminated.clone()
        i = 0
        while (not torch.all(dones)) and (i < self.params.ls_max_iter):
            i += 1

            # Evaluate current alpha
            x_cand, u_cand = self.calc_cadidate_solutions(
                alpha,
                self.prob.states,
                self.prob.controls,
                delta_x,
                delta_u,
            )

            if self.params.ls_function == "filter":
                cost, gamma, uact = self.calc_metrics(x_cand, u_cand)
                # Update successful environments
                cost_improved = cost < self.best_cost
                gamma_improved = gamma < self.best_gamma
                uact_improved = uact < self.best_uact
                # update_mask = (
                #     cost_improved | gamma_improved | uact_improved
                # ) & line_search_done
                update_mask = (cost_improved | gamma_improved | uact_improved) & ~dones
            elif self.params.ls_function == "merit":
                # Merit Function
                phi = self.merit(x_cand, u_cand)
                phi_improved = phi <= self.best_phi
                self.best_phi[phi_improved] = phi[phi_improved]
                update_mask = phi_improved & ~dones

            if update_mask.any():
                for k in range(self.horizon - 1):
                    self.prob.states[k][update_mask] = x_cand[k][update_mask]
                    self.prob.controls[k][update_mask] = u_cand[k][update_mask]
                    self.prob.pi[k + 1][update_mask] = pi[k + 1][update_mask]
                    # self.prob.ni[k][update_mask] = ni[k][update_mask]
                self.prob.states[-1][update_mask] = x_cand[-1][update_mask]
                # Mark environments as finished
                dones[update_mask] = True

            if self.params.ls_function == "filter":
                # Update best cost and gamma
                self.best_cost[cost_improved & ~dones] = cost[cost_improved & ~dones]
                self.best_gamma[gamma_improved & ~dones] = gamma[
                    gamma_improved & ~dones
                ]
                self.best_uact[uact_improved & ~dones] = uact[uact_improved & ~dones]
            elif self.params.ls_function == "merit":
                # Update best merit
                self.best_phi[phi_improved & ~dones] = phi[phi_improved & ~dones]

            # Update alpha for failed environments
            failed_mask = ~update_mask & ~dones
            if failed_mask.any():
                alpha[failed_mask] *= 0.5
        # if not torch.all(dones):
        #     print("Line search failed: ", dones)
        return i, torch.all(dones)

    def check_termination(self, delta_x, delta_u):
        """
        Check the KKT conditions:
        - ||Lx||_inf < eps
        - ||Lu||_inf < eps
        - ||dynamics(x, u) - x_next||_inf < eps
        - ||g(x, u)||_inf < eps

        ! : For the Lagrangian gradient, we only need to include the active constraints
        """
        ## Lagrangian Gradients ##
        Lx, Lu = self.calc_Lx_Lu()

        ## Constraint Violations ##
        states = torch.stack(self.prob.states, dim=0)
        controls = torch.stack(self.prob.controls, dim=0)

        # Dynamics
        dyn = self.dynamics_violation_(states[1:], states[:-1], controls[:])
        dyn_inf = torch.norm(dyn, p=float("inf"), dim=[0, 2])

        # Underactuation
        if self.prob.underactuation is not None:
            uact = self.underactuation_violation_(states[:-1], controls[:])
            uact_inf = torch.norm(uact, p=float("inf"), dim=[0, 2])

        convergence_error = torch.maximum(dyn_inf, uact_inf)

        # terminate_Lx = max_Lx < self.params.sqp_eps
        # terminate_Lu = max_Lu < self.params.sqp_eps
        # print("Max dyn viols: ", max_dyn_viols, ", ", max_uact_viols)
        # self.iter_log["max_dyn_viol"] = torch.norm(all_dyn_viols, p=float("inf")).item()
        # self.iter_log["max_uact_viol"] = torch.norm(
        #     max_uact_viols, p=float("inf")
        # ).item()

        terminate_Lx = inf_norm(Lx) < self.params.sqp_eps
        terminate_Lu = inf_norm(Lu) < self.params.sqp_eps
        terminate_constraints = convergence_error < self.params.sqp_eps
        return torch.stack(
            [
                # terminate_Lx,
                # terminate_Lu,
                terminate_constraints
            ]
        ).all()

    def calc_cadidate_solutions(self, alpha, curr_x, curr_u, delta_x, delta_u):
        x_cand = []
        u_cand = []
        for k in range(self.horizon):
            x_cand.append(curr_x[k] + torch.mul(alpha, delta_x[k]))
            if k < self.horizon - 1:
                u_cand.append(curr_u[k] + torch.mul(alpha, delta_u[k]))
        return x_cand, u_cand

    def calc_metrics(self, x_cand, u_cand):
        # Returns:
        # cost: total cost
        # dyn_inf: ||dynamics violation||_inf
        # uact_inf:  ||underactuation violation||_inf

        states = torch.stack(x_cand, dim=0)
        controls = torch.stack(u_cand, dim=0)

        cost = self.total_cost_(states, controls)
        dyn = self.dynamics_violation_(states[1:], states[:-1], controls[:])
        dyn_inf = torch.norm(dyn, p=float("inf"), dim=[0, 2])

        if self.prob.underactuation is not None:
            uact = self.underactuation_violation_(states[:-1], controls[:])
            uact_inf = torch.norm(uact, p=float("inf"), dim=[0, 2])

        return cost, dyn_inf, uact_inf

    def total_cost_(self, x, u):
        cost = torch.zeros((self.prob.n_batch))
        for k in range(self.horizon - 1):
            # Calculate total trajectory cost
            cost += self.prob.l(k, x[k], u[k])
        # Add final node cost
        cost += self.prob.l(-1, x[-1])

        return cost

    def dynamics_violation_(self, x_next, x, u):
        f = self.prob.dynamics.f
        return x_next - f(x, u, self.prob.dt)

    def underactuation_violation_(self, x, u):
        h = self.prob.underactuation.h
        return h(x, u)

    # Calculate Lagrangian gradients
    def calc_Lx_Lu(self):
        dt = self.prob.dt
        Lx = torch.zeros((self.prob.n_batch, self.prob.nx))
        Lu = torch.zeros((self.prob.n_batch, self.prob.nu))
        for k in range(self.horizon - 1):
            x = self.prob.states[k]
            u = self.prob.controls[k]
            pi_0 = self.prob.pi[k]
            pi_1 = self.prob.pi[k + 1]
            lx = self.prob.lx
            lu = self.prob.lu
            fx = self.prob.dynamics.fx
            fu = self.prob.dynamics.fu
            Lx += lx(k, x, u)
            Lu += lu(k, x, u)
        # Add final node cost
        x_N = self.prob.states[-1]
        Lx += self.prob.lx(-1, x_N)

        return Lx, Lu
