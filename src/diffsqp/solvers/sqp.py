import sys
import time
import torch
from diffsqp.utils.math import mm, mv, inf_norm
from typing import List

from diffsqp.problems import Problem
from diffsqp.solvers import Lqr
from dataclasses import dataclass


@dataclass
class SqpParams:
    qp_solver: str
    n_B: int
    max_iter: int
    eps: float


@dataclass
class SqpIterationLog:
    # QP stuff
    qp_delta_x: List[torch.Tensor]
    qp_delta_u: List[torch.Tensor]
    qp_pi: List[torch.Tensor]
    qp_delta_nu: List[torch.Tensor]

    # Logging
    sqp_iterations: int
    termination_time_s: int
    cuda_reserved_bytes: int
    cuda_allocated_bytes: int


class Sqp:
    def __init__(self, prob: Problem, params: SqpParams) -> None:
        self.prob = prob
        self.params = params
        self.horizon = self.prob.horizon

        # self.admm_solver = Admm(self.prob, qp_solver)
        self.qp_solver = None
        if self.params.qp_solver == "lqr":
            self.qp_solver = Lqr(prob)

        # Log best line search metrics
        self.best_cost, self.best_gamma, self.best_uact = self.calc_metrics(
            self.prob.states,
            self.prob.controls,
        )

        self.terminated = torch.zeros((self.params.n_B), dtype=torch.bool)

        self.iter_log = {
            "nB": self.params.n_B,
            "terminated": 0,
            "ssqp_iterations": 0,
            "t_solve_s": 0.0,
            "cuda_reserved_bytes": 0,
            "cuda_allocated_bytes": 0,
            "t_qp_solve": [0.0] * self.params.max_iter,
            "t_line_search": [0.0] * self.params.max_iter,
            "line_search_iters": [0.0] * self.params.max_iter,
            "max_dyn_viol": 0.0,
            "max_uact_viol": 0.0,
        }

    def solve(self):
        # Solve for max_iter steps
        t_solve_start = time.time()
        for iter in range(self.params.max_iter):
            # Get LQR corrections
            # Perform ADMM step
            start = time.time()
            delta_x_qp, delta_u_qp, pi_qp, ni_qp = self.qp_solver.solve()
            end = time.time()
            t_qp_solve = end - start
            self.iter_log["t_qp_solve"] = t_qp_solve

            # Line search
            start = time.time()
            alpha, dones, ls_iters = self.line_search(
                delta_x_qp, delta_u_qp, pi_qp, ni_qp
            )
            end = time.time()
            t_line_search = end - start
            self.iter_log["t_line_search"] = t_line_search
            self.iter_log["line_search_iters"] = ls_iters

            # Check termination
            if self.check_termination(delta_x_qp, delta_u_qp):
                t_solve_end = time.time()
                break
        print("SSQP Total Iterations: ", iter + 1)
        print(
            "Terminated Environments: ",
            torch.count_nonzero(self.terminated).item(),
            "/",
            self.terminated.shape[0],
        )

        # Fill log
        self.iter_log["t_solve_s"] = t_solve_end - t_solve_start
        self.iter_log["ssqp_iterations"] = iter + 1
        self.iter_log["terminated"] = torch.count_nonzero(self.terminated).item()
        if torch.get_default_device() != "cpu":
            self.iter_log["cuda_reserved_bytes"] = torch.cuda.memory_reserved(0)
            self.iter_log["cuda_allocated_bytes"] = torch.cuda.memory_allocated(0)
        return self.iter_log

    def line_search(self, delta_x, delta_u, pi, ni, max_iter: float = 10):
        alpha = torch.ones((self.params.n_B, 1))
        dones = self.terminated.clone()
        i = 0
        while (not torch.all(dones)) and (i < max_iter):
            i += 1
            gamma = torch.zeros((self.params.n_B, 1))

            # Evaluate current alpha
            x_cand, u_cand = self.calc_cadidate_solutions(
                alpha,
                self.prob.states,
                self.prob.controls,
                delta_x,
                delta_u,
            )
            cost, gamma, uact = self.calc_metrics(x_cand, u_cand)

            # Update successful environments
            cost_improved = cost < self.best_cost
            gamma_improved = gamma < self.best_gamma
            uact_improved = uact < self.best_uact
            # update_mask = (
            #     cost_improved | gamma_improved | uact_improved
            # ) & line_search_done
            update_mask = (cost_improved | gamma_improved | uact_improved) & ~dones

            # Merit Function
            # merit = cost + self.beta * gamma + self.beta * uact
            # merit_improved = merit < self.best_merit
            # update_mask = merit_improved & ~dones
            # print(merit_improved)

            if update_mask.any():
                for k in range(self.horizon - 1):
                    self.prob.states[k][update_mask] = x_cand[k][update_mask]
                    self.prob.controls[k][update_mask] = u_cand[k][update_mask]
                    self.prob.pi[k + 1][update_mask] = pi[k + 1][update_mask]
                    # self.prob.ni[k][update_mask] = ni[k][update_mask]
                self.prob.states[-1][update_mask] = x_cand[-1][update_mask]
                # Mark environments as finished
                dones[update_mask] = True

            # Update best cost and gamma
            self.best_cost[cost_improved & ~dones] = cost[cost_improved & ~dones]
            self.best_gamma[gamma_improved & ~dones] = gamma[gamma_improved & ~dones]
            self.best_uact[uact_improved & ~dones] = uact[uact_improved & ~dones]

            # Update alpha for failed environments
            failed_mask = ~update_mask & ~dones
            if failed_mask.any():
                alpha[failed_mask] *= 0.5
        # if not torch.all(dones):
        #     print("Line search failed: ", dones)
        return alpha, dones, i

    def check_termination(self, delta_x, delta_u):
        ## Lagrangian Gradient ##
        # Lx, Lu = self.calc_Lx_Lu()
        # max_Lx = torch.norm(Lx, p=float("inf"), dim=1)
        # max_Lu = torch.norm(Lx, p=float("inf"), dim=1)

        max = torch.zeros(self.params.n_B)
        for i in range(len(delta_u)):
            # cand = torch.max(torch.square(dx), dim=1).values
            dx = delta_x[i]
            du = delta_u[i]
            res = torch.max(
                mm(dx.unsqueeze(2).transpose(1, 2), dx.unsqueeze(2)).squeeze(2)
                + mm(du.unsqueeze(2).transpose(1, 2), du.unsqueeze(2)).squeeze(2),
                dim=1,
            ).values
            max[res > max] = res[res > max]
        dx = delta_x[i]
        du = delta_u[i]
        res = torch.max(
            mm(dx.unsqueeze(2).transpose(1, 2), dx.unsqueeze(2)).squeeze(2), dim=1
        ).values
        max[res > max] = res[res > max]
        dx_crit = max < 0.1

        ## Dynamics Violation ##
        # Track largest violation for each batch
        max_dyn_viols = torch.zeros(self.params.n_B)
        max_uact_viols = torch.zeros(self.params.n_B)
        for k in range(self.horizon - 1):
            x0 = self.prob.states[k]
            u0 = self.prob.controls[k]
            x1 = self.prob.states[k + 1]
            f = self.prob.dynamics[k].f

            dyn_viol = self.calc_dynamics_violation(f, x1, x0, u0)
            dyn_inf_norm = torch.norm(dyn_viol, p=float("inf"), dim=1)
            index_mask = dyn_inf_norm > max_dyn_viols
            max_dyn_viols[index_mask] = dyn_inf_norm[index_mask]

            if self.prob.constraints[k]:
                uact_viol = self.calc_underactuation_violation(k, x0, u0)
                uact_inf_norm = inf_norm(uact_viol)
                index_mask = uact_inf_norm > max_uact_viols
                max_uact_viols[index_mask] = uact_inf_norm[index_mask]

        # terminate_Lx = max_Lx < self.params.eps
        # terminate_Lu = max_Lu < self.params.eps
        # print("Max dyn viols: ", max_dyn_viols, ", ", max_uact_viols)
        self.iter_log["max_dyn_viol"] = torch.norm(max_dyn_viols, p=float("inf")).item()
        self.iter_log["max_uact_viol"] = torch.norm(
            max_uact_viols, p=float("inf")
        ).item()
        terminate_dyn_viols = max_dyn_viols < self.params.eps
        terminate_uact_viols = max_uact_viols < self.params.eps

        # self.terminated = torch.logical_or(terminate_Lx, terminate_Lu)
        self.terminated = torch.logical_and(terminate_dyn_viols, terminate_uact_viols)
        self.terminated = torch.logical_and(self.terminated, dx_crit)

    def calc_cadidate_solutions(self, alpha, curr_x, curr_u, delta_x, delta_u):
        x_cand = []
        u_cand = []
        for k in range(self.horizon):
            x_cand.append(curr_x[k] + torch.mul(alpha, delta_x[k]))
            if k < self.horizon - 1:
                u_cand.append(curr_u[k] + torch.mul(alpha, delta_u[k]))
        return x_cand, u_cand

    def calc_metrics(self, x_cand, u_cand):
        cost = torch.zeros((self.params.n_B))
        gamma = torch.zeros((self.params.n_B))
        uact = torch.zeros((self.params.n_B))
        for k in range(self.horizon - 1):
            # Calculate total trajectory cost
            cost += self.prob.l(k, x_cand[k], u_cand[k])
            # Calculate constraint violations
            dyn_viol = self.calc_dynamics_violation(
                self.prob.dynamics[k].f,
                x_cand[k + 1],
                x_cand[k],
                u_cand[k],
            )
            gamma += inf_norm(dyn_viol)
            if self.prob.constraints[k]:
                uact_viol = self.calc_underactuation_violation(
                    k,
                    x_cand[k],
                    u_cand[k],
                )
                uact += inf_norm(uact_viol)
        # Add final node cost
        cost += self.prob.l(-1, x_cand[-1])
        return cost, gamma, uact

    def calc_dynamics_violation(self, f, x_next, x, u):
        return x_next - f(x, u, self.prob.dt)

    def calc_underactuation_violation(self, stage_idx, x, u):
        return self.prob.g(stage_idx, x, u)

    # Calculate Lagrangian gradients
    def calc_Lx_Lu(self):
        dt = self.prob.dt
        Lx = torch.zeros((self.params.n_B, self.prob.nx))
        Lu = torch.zeros((self.params.n_B, self.prob.nu))
        for k in range(self.horizon - 1):
            x = self.prob.states[k]
            u = self.prob.controls[k]
            pi_0 = self.prob.pi[k]
            pi_1 = self.prob.pi[k + 1]
            lx = self.prob.lx
            lu = self.prob.lu
            fx = self.prob.dynamics[k].fx
            fu = self.prob.dynamics[k].fu
            Lx += lx(k, x, u) + mv(torch.transpose(fx(x, u, dt), 1, 2), pi)
            Lu += lu(k, x, u) + mv(torch.transpose(fu(x, u, dt), 1, 2), pi)
        # Add final node cost
        x_N = self.prob.states[-1]
        Lx += self.prob.lx(-1, x_N)

        return Lx, Lu
