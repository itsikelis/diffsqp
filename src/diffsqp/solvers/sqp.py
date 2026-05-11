import sys
import time
import torch
from diffsqp.utils.math import mm, mv, inf_norm
from typing import List

from diffsqp.problems import Problem
from diffsqp.solvers import Lqr, QP
from dataclasses import dataclass


@dataclass
class SqpParams:
    qp_solver: str
    ls_technique: str
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

        if self.params.ls_technique == "filter":
            # Log best line search metrics
            self.best_cost, self.best_gamma, self.best_uact = self.calc_metrics(
                self.prob.states,
                self.prob.controls,
            )
        elif self.params.ls_technique == "merit":
            # Merit function
            self.ls_mu = 1e6
            self.best_phi = self.merit(self.prob.states, self.prob.controls)

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
            # TODO: Log ls time and total iters
            alpha, dones, ls_iters = self.line_search(
                delta_x_qp, delta_u_qp, pi_qp, ni_qp
            )

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

    def merit(self, x, u):
        J, dyn, uact = self.calc_metrics(x, u)
        # idx = uact > dyn
        # dyn[idx] = uact[idx]
        return J + self.ls_mu * dyn + self.ls_mu * uact

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

            if self.params.ls_technique == "filter":
                cost, gamma, uact = self.calc_metrics(x_cand, u_cand)
                # Update successful environments
                cost_improved = cost < self.best_cost
                gamma_improved = gamma < self.best_gamma
                uact_improved = uact < self.best_uact
                # update_mask = (
                #     cost_improved | gamma_improved | uact_improved
                # ) & line_search_done
                update_mask = (cost_improved | gamma_improved | uact_improved) & ~dones
            elif self.params.ls_technique == "merit":
                # Merit Function
                phi = self.merit(x_cand, u_cand)
                phi_improved = phi <= self.best_phi
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

            if self.params.ls_technique == "filter":
                # Update best cost and gamma
                self.best_cost[cost_improved & ~dones] = cost[cost_improved & ~dones]
                self.best_gamma[gamma_improved & ~dones] = gamma[
                    gamma_improved & ~dones
                ]
                self.best_uact[uact_improved & ~dones] = uact[uact_improved & ~dones]
            elif self.params.ls_technique == "merit":
                # Update best merit
                self.best_phi[phi_improved & ~dones] = phi[phi_improved & ~dones]

            # Update alpha for failed environments
            failed_mask = ~update_mask & ~dones
            if failed_mask.any():
                alpha[failed_mask] *= 0.5
        # if not torch.all(dones):
        #     print("Line search failed: ", dones)
        return alpha, dones, i

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
        # print(Lx)

        ## Constraint Violations ##
        # Dynamics
        states_tensor = torch.stack(self.prob.states, dim=0)
        controls_tensor = torch.stack(self.prob.controls, dim=0)
        x_curr = states_tensor[:-1]
        u_curr = controls_tensor[:]
        x_next = states_tensor[1:]
        f = self.prob.dynamics[0].f
        all_dyn_viols = x_next - f(x_curr, u_curr, self.prob.dt)

        # Underactuation
        max_uact_viols = torch.zeros(self.params.n_B)
        for k in range(self.horizon - 1):
            x0 = self.prob.states[k]
            u0 = self.prob.controls[k]
            x1 = self.prob.states[k + 1]
            if self.prob.constraints[k]:
                uact_viol = self.underactuation_violation(k, x0, u0)
                uact_inf_norm = inf_norm(uact_viol)
                index_mask = uact_inf_norm > max_uact_viols
                max_uact_viols[index_mask] = uact_inf_norm[index_mask]

        # terminate_Lx = max_Lx < self.params.eps
        # terminate_Lu = max_Lu < self.params.eps
        # print("Max dyn viols: ", max_dyn_viols, ", ", max_uact_viols)
        self.iter_log["max_dyn_viol"] = torch.norm(all_dyn_viols, p=float("inf")).item()
        self.iter_log["max_uact_viol"] = torch.norm(
            max_uact_viols, p=float("inf")
        ).item()

        terminate_Lx = inf_norm(Lx) < self.params.eps
        terminate_Lu = inf_norm(Lu) < self.params.eps
        terminate_dyn_viols = (
            torch.norm(all_dyn_viols, p=float("inf"), dim=[0, 2]) < self.params.eps
        )
        terminate_uact_viols = max_uact_viols < self.params.eps
        return torch.stack(
            [
                # terminate_Lx,
                # terminate_Lu,
                terminate_dyn_viols,
                terminate_uact_viols,
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
        # J: total cost
        # dyn: ||dynamics violation||_inf
        # uact:  ||underactuation violation||_inf
        J = torch.zeros((self.params.n_B))
        dyn = torch.zeros((self.params.n_B))
        uact = torch.zeros((self.params.n_B))
        for k in range(self.horizon - 1):
            # Calculate total trajectory cost
            J += self.prob.l(k, x_cand[k], u_cand[k])

            # Dynamics violations
            dyn_cand = inf_norm(
                self.stage_dynamics_violation(
                    self.prob.dynamics[k].f,
                    x_cand[k + 1],
                    x_cand[k],
                    u_cand[k],
                )
            )
            # Keep biggest violations of all envs across the horizon
            idx = [dyn_cand > dyn]
            dyn[tuple(idx)] = dyn_cand[tuple(idx)]

            # Underactuation violations
            if self.prob.constraints[k]:
                uact_cand = inf_norm(
                    self.underactuation_violation(
                        k,
                        x_cand[k],
                        u_cand[k],
                    )
                )
                # Keep biggest violations of all envs across the horizon
                idx = uact_cand > uact
                uact[idx] = uact_cand[idx]

        # Add final node cost
        J += self.prob.l(-1, x_cand[-1])
        return J, dyn, uact

    def stage_dynamics_violation(self, f, x_next, x, u):
        return x_next - f(x, u, self.prob.dt)

    def underactuation_violation(self, stage_idx, x, u):
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
            Lx += lx(k, x, u)
            Lu += lu(k, x, u)
        # Add final node cost
        x_N = self.prob.states[-1]
        Lx += self.prob.lx(-1, x_N)

        return Lx, Lu
